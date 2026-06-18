"""Small COCO utilities for open cold-start selection."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def load_coco(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path | str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def image_key(file_name: str, mode: str = "basename") -> str:
    if mode == "basename":
        return Path(file_name).name
    if mode == "relative":
        return str(file_name).replace("\\", "/")
    raise ValueError(f"Unsupported image key mode: {mode}")


def coco_image_keys(coco: dict, *, key_mode: str = "basename") -> set[str]:
    return {image_key(str(image["file_name"]), key_mode) for image in coco.get("images", [])}


def target_counts_by_image_key(
    coco: dict,
    target_class_id: int,
    *,
    key_mode: str = "basename",
) -> dict[str, int]:
    id_to_key = {
        int(image["id"]): image_key(str(image["file_name"]), key_mode)
        for image in coco.get("images", [])
    }
    counts: dict[str, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) == int(target_class_id):
            key = id_to_key.get(int(ann["image_id"]))
            if key is not None:
                counts[key] += 1
    return dict(counts)


def subset_coco(coco: dict, selected_keys: Iterable[str], *, key_mode: str = "basename") -> dict:
    selected = set(selected_keys)
    source_id_to_key = {
        int(image["id"]): image_key(str(image["file_name"]), key_mode)
        for image in coco.get("images", [])
    }

    images = []
    old_to_new_id = {}
    for new_id, image in enumerate(coco.get("images", []), start=1):
        key = image_key(str(image["file_name"]), key_mode)
        if key not in selected:
            continue
        old_to_new_id[int(image["id"])] = len(images) + 1
        new_image = dict(image)
        new_image["id"] = len(images) + 1
        images.append(new_image)

    annotations = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        key = source_id_to_key.get(image_id)
        if key not in selected:
            continue
        new_ann = dict(ann)
        new_ann["id"] = len(annotations) + 1
        new_ann["image_id"] = old_to_new_id[image_id]
        annotations.append(new_ann)

    if len(images) != len(selected):
        available = {image_key(str(image["file_name"]), key_mode) for image in coco.get("images", [])}
        missing = sorted(selected - available)
        raise ValueError(f"COCO file is missing {len(missing)} selected images: {missing[:5]}")

    return {
        "images": images,
        "annotations": annotations,
        "categories": list(coco.get("categories", [])),
    }


def class_counts(coco: dict) -> dict[str, int]:
    counts = Counter(int(ann["category_id"]) for ann in coco.get("annotations", []))
    return {str(k): int(v) for k, v in sorted(counts.items())}
