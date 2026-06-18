#!/usr/bin/env python
"""Train frozen-DINOv3 linear verifier heads from COCO ROI crops."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_key_values(items: list[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in items:
        key, value = item.split("=", 1)
        out[int(key)] = float(value)
    return out


def parse_int_map(items: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for item in items:
        key, value = item.split("=", 1)
        out[int(key)] = int(value)
    return out


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def infer_category_map(
    cocos: list[dict],
    *,
    num_classes: int,
    category_id_base: str,
    manual_items: list[str],
) -> dict[int, int]:
    if manual_items:
        mapping = parse_int_map(manual_items)
    else:
        raw_ids = sorted(
            {
                int(ann["category_id"])
                for coco in cocos
                for ann in coco.get("annotations", [])
            }
        )
        if not raw_ids:
            raise ValueError("No annotations found for verifier training.")
        if category_id_base == "zero" or (
            category_id_base == "auto" and all(0 <= cid < num_classes for cid in raw_ids)
        ):
            mapping = {cid: cid for cid in raw_ids}
        elif category_id_base == "one" or (
            category_id_base == "auto" and all(1 <= cid <= num_classes for cid in raw_ids)
        ):
            mapping = {cid: cid - 1 for cid in raw_ids}
        else:
            raise ValueError(
                "Cannot infer category mapping. Use --category-id-map raw_id=internal_id "
                "or choose --category-id-base zero/one."
            )
    bad = {raw: mapped for raw, mapped in mapping.items() if mapped < 0 or mapped >= num_classes}
    if bad:
        raise ValueError(f"Mapped category ids outside [0,{num_classes - 1}]: {bad}")
    return mapping


def xywh_to_xyxy(box: list[float], width: int, height: int, expand: float) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w / 2.0, y + h / 2.0
    ew, eh = w * (1.0 + 2.0 * expand), h * (1.0 + 2.0 * expand)
    x1 = max(0, int(round(cx - ew / 2.0)))
    y1 = max(0, int(round(cy - eh / 2.0)))
    x2 = min(width, int(round(cx + ew / 2.0)))
    y2 = min(height, int(round(cy + eh / 2.0)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box after expansion: {box}")
    return x1, y1, x2, y2


def box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def collect_positive_records(
    coco: dict,
    image_dir: Path,
    expand: float,
    category_map: dict[int, int],
) -> list[dict]:
    images = {int(image["id"]): image for image in coco.get("images", [])}
    rows = []
    for ann in coco.get("annotations", []):
        image = images[int(ann["image_id"])]
        path = image_dir / image["file_name"]
        raw_category_id = int(ann["category_id"])
        if raw_category_id not in category_map:
            raise ValueError(f"Missing category mapping for raw category_id={raw_category_id}")
        rows.append(
            {
                "image_path": str(path),
                "bbox": ann["bbox"],
                "raw_category_id": raw_category_id,
                "category_id": int(category_map[raw_category_id]),
                "objectness": 1,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "expand": float(expand),
            }
        )
    return rows


def add_background_records(
    rows: list[dict],
    coco: dict,
    image_dir: Path,
    *,
    negatives_per_image: int,
    iou_max: float,
    seed: int,
) -> list[dict]:
    if negatives_per_image <= 0:
        return []
    rng = random.Random(seed)
    anns_by_image: dict[int, list[tuple[float, float, float, float]]] = {}
    for ann in coco.get("annotations", []):
        x, y, w, h = [float(v) for v in ann["bbox"]]
        anns_by_image.setdefault(int(ann["image_id"]), []).append((x, y, x + w, y + h))

    negatives = []
    for image in coco.get("images", []):
        width, height = int(image["width"]), int(image["height"])
        gt_boxes = anns_by_image.get(int(image["id"]), [])
        if not gt_boxes:
            continue
        sizes = [(b[2] - b[0], b[3] - b[1]) for b in gt_boxes]
        tries = 0
        while len([r for r in negatives if r["image_path"].endswith(image["file_name"])]) < negatives_per_image:
            tries += 1
            if tries > 200:
                break
            bw, bh = rng.choice(sizes)
            bw = min(width, max(8.0, bw * rng.uniform(0.7, 1.4)))
            bh = min(height, max(8.0, bh * rng.uniform(0.7, 1.4)))
            x1 = rng.uniform(0, max(1.0, width - bw))
            y1 = rng.uniform(0, max(1.0, height - bh))
            cand = (x1, y1, x1 + bw, y1 + bh)
            if max((box_iou_xyxy(cand, gt) for gt in gt_boxes), default=0.0) > iou_max:
                continue
            negatives.append(
                {
                    "image_path": str(image_dir / image["file_name"]),
                    "bbox": [x1, y1, bw, bh],
                    "category_id": -1,
                    "objectness": 0,
                    "width": width,
                    "height": height,
                    "expand": 0.0,
                }
            )
    rows.extend(negatives)
    return negatives


class CropDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int):
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[idx]
        path = Path(row["image_path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            x1, y1, x2, y2 = xywh_to_xyxy(row["bbox"], row["width"], row["height"], row["expand"])
            crop = image.crop((x1, y1, x2, y2))
            crop = TF.resize(crop, [self.image_size, self.image_size], antialias=True)
            tensor = TF.normalize(TF.to_tensor(crop), IMAGENET_MEAN, IMAGENET_STD)
        return tensor, int(row["category_id"]), int(row["objectness"])


def extract_features(rows: list[dict], model_dir: Path, image_size: int, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise SystemExit("Install transformers to train verifier heads.") from exc

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = AutoModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    loader = DataLoader(CropDataset(rows, image_size), batch_size=batch_size, shuffle=False, num_workers=0)
    feats, cls_labels, obj_labels = [], [], []
    with torch.inference_mode():
        for images, cls, obj in loader:
            images = images.to(device)
            out = model(pixel_values=images).last_hidden_state[:, 0, :].float()
            out = F.normalize(out, dim=1)
            feats.append(out.cpu().numpy())
            cls_labels.extend(int(v) for v in cls.tolist())
            obj_labels.extend(int(v) for v in obj.tolist())
    return np.concatenate(feats, axis=0), np.asarray(cls_labels), np.asarray(obj_labels)


def fit_logistic(features: np.ndarray, labels: np.ndarray, class_weight: dict | str | None = None):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise SystemExit("Install scikit-learn to train verifier heads.") from exc

    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(features, labels)
    return clf


def require_classes(clf, expected: list[int], name: str) -> None:
    actual = [int(v) for v in clf.classes_.tolist()]
    if actual != expected:
        raise ValueError(f"{name} classes are {actual}, expected {expected}.")


def calibrate_thresholds(probs: np.ndarray, labels: np.ndarray, target_precision: float) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    pred = probs.argmax(axis=1)
    for cls in range(probs.shape[1]):
        scores = probs[:, cls]
        candidates = sorted(set(float(v) for v in scores[pred == cls]), reverse=True)
        chosen = 0.5
        for thr in reversed(candidates):
            keep = (pred == cls) & (scores >= thr)
            if keep.sum() == 0:
                continue
            precision = float((labels[keep] == cls).mean())
            if precision >= target_precision:
                chosen = float(thr)
                break
        thresholds[int(cls)] = chosen
    return thresholds


def confusion_matrix(pred: np.ndarray, labels: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, guess in zip(labels, pred):
        if 0 <= int(truth) < num_classes and 0 <= int(guess) < num_classes:
            matrix[int(truth), int(guess)] += 1
    return matrix.tolist()


def save_linear_head(path: Path, clf, *, num_classes: int | None = None, metadata: dict | None = None) -> None:
    coef = torch.as_tensor(clf.coef_, dtype=torch.float32)
    bias = torch.as_tensor(clf.intercept_, dtype=torch.float32)
    if coef.shape[0] == 1 and num_classes == 2:
        payload = {"weight": coef, "bias": bias, "in_dim": int(coef.shape[1])}
    else:
        payload = {
            "weight": coef,
            "bias": bias,
            "in_dim": int(coef.shape[1]),
            "num_classes": int(num_classes or coef.shape[0]),
        }
    if metadata:
        payload.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-coco", type=Path, required=True)
    parser.add_argument("--train-image-dir", type=Path, required=True)
    parser.add_argument("--val-coco", type=Path, required=True)
    parser.add_argument("--val-image-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("weights/pretrained/dinov3_vits16"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verifier"))
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi-expand", type=float, default=0.15)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--class-weight", action="append", default=[], help="Optional class weight, e.g. 0=1.5")
    parser.add_argument(
        "--category-id-base",
        choices=["auto", "zero", "one"],
        default="auto",
        help="Map raw COCO category_id values to internal detector labels.",
    )
    parser.add_argument(
        "--category-id-map",
        action="append",
        default=[],
        help="Explicit raw-to-internal class mapping, e.g. 1=0. May be repeated.",
    )
    parser.add_argument("--negative-per-image", type=int, default=3)
    parser.add_argument("--negative-iou-max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_coco = load_coco(args.train_coco)
    val_coco = load_coco(args.val_coco)
    category_map = infer_category_map(
        [train_coco, val_coco],
        num_classes=args.num_classes,
        category_id_base=args.category_id_base,
        manual_items=args.category_id_map,
    )
    train_rows = collect_positive_records(train_coco, args.train_image_dir, args.roi_expand, category_map)
    val_rows = collect_positive_records(val_coco, args.val_image_dir, args.roi_expand, category_map)
    train_neg = add_background_records(
        train_rows,
        train_coco,
        args.train_image_dir,
        negatives_per_image=args.negative_per_image,
        iou_max=args.negative_iou_max,
        seed=args.seed,
    )
    val_neg = add_background_records(
        val_rows,
        val_coco,
        args.val_image_dir,
        negatives_per_image=max(1, args.negative_per_image),
        iou_max=args.negative_iou_max,
        seed=args.seed + 1,
    )

    x_train, y_train_cls, y_train_obj = extract_features(
        train_rows, args.model_dir, args.image_size, args.batch_size, args.device
    )
    x_val, y_val_cls, y_val_obj = extract_features(
        val_rows, args.model_dir, args.image_size, args.batch_size, args.device
    )

    class_weights = parse_key_values(args.class_weight)
    weights = {i: float(class_weights.get(i, 1.0)) for i in range(args.num_classes)}
    pos_train = y_train_cls >= 0
    pos_val = y_val_cls >= 0
    clf4 = fit_logistic(x_train[pos_train], y_train_cls[pos_train], class_weight=weights)
    require_classes(clf4, list(range(args.num_classes)), "four-class verifier")
    probs4 = clf4.predict_proba(x_val[pos_val])
    pred4 = probs4.argmax(axis=1)
    thresholds = calibrate_thresholds(probs4, y_val_cls[pos_val], args.target_precision)
    acc4 = float((pred4 == y_val_cls[pos_val]).mean()) if pos_val.any() else 0.0

    head4_path = args.output_dir / "verifier_head_4cls.pt"
    save_linear_head(
        head4_path,
        clf4,
        num_classes=args.num_classes,
        metadata={
            "accept_threshold": thresholds,
            "class_loss_scale": weights,
            "feature_model_path": str(args.model_dir),
            "roi_expand": float(args.roi_expand),
            "l2_normalized": True,
        },
    )

    binary_path = args.output_dir / "verifier_head.pt"
    if len(np.unique(y_train_obj)) == 2:
        clf_bin = fit_logistic(x_train, y_train_obj, class_weight="balanced")
        require_classes(clf_bin, [0, 1], "binary verifier")
        save_linear_head(binary_path, clf_bin, num_classes=2)
        obj_prob = clf_bin.predict_proba(x_val)[:, 1]
        obj_pred = (obj_prob >= 0.5).astype(int)
        obj_acc = float((obj_pred == y_val_obj).mean())
    else:
        obj_acc = None

    summary = {
        "train_coco": str(args.train_coco),
        "val_coco": str(args.val_coco),
        "model_dir": str(args.model_dir),
        "train_positive_crops": int(pos_train.sum()),
        "val_positive_crops": int(pos_val.sum()),
        "train_negative_crops": int(len(train_neg)),
        "val_negative_crops": int(len(val_neg)),
        "category_id_map": {str(k): int(v) for k, v in sorted(category_map.items())},
        "class_counts": {str(k): int(v) for k, v in Counter(y_train_cls[pos_train]).items()},
        "head4_path": str(head4_path),
        "binary_head_path": str(binary_path) if binary_path.exists() else None,
        "val_macro_accuracy": acc4,
        "val_confusion": confusion_matrix(pred4, y_val_cls[pos_val], args.num_classes),
        "accept_threshold": thresholds,
        "target_precision": float(args.target_precision),
        "objectness_accuracy_at_0_5": obj_acc,
    }
    (args.output_dir / "verifier_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
