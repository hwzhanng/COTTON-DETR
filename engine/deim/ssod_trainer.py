"""SoftTeacher-style SSOD trainer for DEIMv2."""

import random
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.ssod_augmentation import SSODPairAugmentor, apply_mic_mask
from ..misc import dist_utils
from ..backbone.dinov3_backbone import DINOv3Backbone
from .adaptive_threshold import ClassAdaptivePseudoThreshold
from .box_ops import box_cxcywh_to_xyxy
from .cross_view_consistency import compute_query_consistency
from .pseudo_label_mining import cost_based_pseudo_mining, _prediction_view


def _move_batch(batch, device: torch.device):
    samples, targets = batch
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
    return samples, targets


def _strip_aux_target_fields(targets: Iterable[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    stripped = []
    for target in targets:
        stripped.append({"boxes": target["boxes"], "labels": target["labels"]})
    return stripped


def _merge_targets(
    high_targets: List[Dict[str, torch.Tensor]],
    mined_targets: List[Dict[str, torch.Tensor]],
) -> List[Dict[str, torch.Tensor]]:
    merged = []
    for high, mined in zip(high_targets, mined_targets):
        if mined["labels"].numel() == 0:
            merged.append(high)
            continue
        if high["labels"].numel() == 0:
            merged.append(mined)
            continue
        merged.append({
            "boxes": torch.cat([high["boxes"], mined["boxes"]], dim=0),
            "labels": torch.cat([high["labels"], mined["labels"]], dim=0),
            "scores": torch.cat([high["scores"], mined["scores"]], dim=0),
            "weights": torch.cat([high["weights"], mined["weights"]], dim=0),
        })
    return merged


def _target_count(targets: Iterable[Dict[str, torch.Tensor]]) -> int:
    return sum(int(target["labels"].numel()) for target in targets)


def _target_class_count(targets: Iterable[Dict[str, torch.Tensor]], class_id: int) -> int:
    total = 0
    for target in targets:
        labels = target.get("labels")
        if labels is not None and labels.numel() > 0:
            total += int((labels.long() == int(class_id)).sum().item())
    return total


def _target_score_mean(targets: Iterable[Dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    scores = []
    for target in targets:
        value = target.get("scores")
        if value is not None and value.numel() > 0:
            scores.append(value.detach())
    if not scores:
        return torch.zeros((), device=device)
    return torch.cat(scores).mean()


def _target_flag_sum(targets: Iterable[Dict[str, torch.Tensor]], key: str, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for target in targets:
        value = target.get(key)
        if value is not None:
            total = total + value.to(device=device, dtype=torch.float32).sum()
    return total


def _assert_finite_outputs(outputs: Dict[str, torch.Tensor], prefix: str) -> None:
    for name in ("pred_logits", "pred_boxes"):
        value = outputs.get(name)
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(f"{prefix}.{name} contains NaN or Inf.")


class SSODTrainer:
    def __init__(
        self,
        student_model: nn.Module,
        criterion: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        ema_decay: float = 0.9996,
        pseudo_threshold: float = 0.7,
        loss_weights: Optional[Dict[str, float]] = None,
        mining_cfg: Optional[Dict] = None,
        consistency_cfg: Optional[Dict] = None,
        augmentation_cfg: Optional[Dict] = None,
        adaptive_threshold: Optional[ClassAdaptivePseudoThreshold] = None,
        num_classes: Optional[int] = None,
        ema_decay_warmup_start: float = 0.99,
        ema_decay_warmup_epochs: int = 0,
        ema_decay_burnin: Optional[float] = None,
        ema_decay_warmup_curve: str = "cosine",
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.5,
        mic_prob: float = 0.0,
        mic_patch_size: int = 32,
        mic_mask_ratio: float = 0.5,
        mic_fill_value: Optional[Sequence[float]] = None,
        pseudo_dn_cfg: Optional[Dict] = None,
        distill_cfg: Optional[Dict] = None,
        verifier_cfg: Optional[Dict] = None,
        soft_bg_cfg: Optional[Dict] = None,
    ) -> None:
        self.student = student_model
        self.teacher = teacher_model if teacher_model is not None else deepcopy(dist_utils.de_parallel(student_model))
        self.criterion = criterion
        self.ema_decay = float(ema_decay)
        self.pseudo_threshold = pseudo_threshold
        self.loss_weights = {
            "supervised": 1.0,
            "unsupervised": 1.0,
            "synthetic": 0.5,
            "consistency": 0.3,
            # REDESIGN: frozen-DINOv3 dense feature distillation on the
            # unlabeled stream. 0.0 default => absent config is a no-op.
            "distill": 0.0,
        }
        if loss_weights:
            self.loss_weights.update(loss_weights)
        # The mining function previously double-scaled mined targets (per-target
        # weight * mined_weight, then per-loss * mined_loss_weight). We now keep
        # mined_weight purely as the trainer-level loss scalar; the mining
        # function ignores its ``mined_weight`` argument for per-target weights.
        self.mining_cfg = dict(mining_cfg or {})
        self.use_mined_as_pseudo_loss = bool(self.mining_cfg.pop("use_mined_as_pseudo_loss", False))
        self.mined_loss_weight = float(self.mining_cfg.get("mined_weight", 0.5))
        # A. mining revival: scale factor that produces looser candidate
        # thresholds from the high-confidence ones. With scale=1.0 candidate
        # equals high and mining contributes nothing (fix6 measured 0.02%
        # acceptance). scale<1.0 lets candidate be a strict superset, so
        # gmm_idx can find labels not in high_idx. Popped here so it is not
        # forwarded as a positional kwarg to the mining function.
        # Default 0.5 is intentionally aggressive: with scale=0.7 the GMM cost
        # filter inside mining still rejects nearly everything (verified in
        # unit test). 0.5 expands the candidate pool to ~2-3x and lets mining
        # surface a small number of mined labels per step in typical score distributions.
        self.candidate_threshold_scale = float(self.mining_cfg.pop("candidate_threshold_scale", 0.5))
        self.consistency_cfg = consistency_cfg or {}
        self.augmentor = SSODPairAugmentor(**(augmentation_cfg or {}))
        self.current_epoch = 0
        self.num_classes = int(num_classes) if num_classes is not None else None

        # T1: per-class adaptive thresholds (LabelMatch style).
        self.adaptive_threshold = adaptive_threshold
        if self.adaptive_threshold is not None and self.num_classes is None:
            self.num_classes = int(self.adaptive_threshold.num_classes)

        # T4 + T-extra: EMA decay scheduling.
        # ``ema_decay_target``: long-run SSOD decay (e.g. 0.9996).
        # ``ema_decay_burnin``: separate decay used during burn-in. The legacy
        # behaviour (0.9996 throughout burn-in) made the teacher lag the
        # student so badly that the EMA "caught up" the instant SSOD started,
        # creating a fake ~+0.024 AP jump that masked SSOD's true contribution.
        # Setting this to a small value like 0.99 keeps the teacher tracking
        # student during burn-in, so val curves are interpretable.
        # ``ema_decay_warmup_*``: ramp at SSOD start; cosine avoids a sharp
        # discontinuity that can spike loss for one epoch (Mean Teacher).
        self.ema_decay_target = float(ema_decay)
        self.ema_decay_burnin = float(ema_decay_burnin) if ema_decay_burnin is not None else None
        self.ema_decay_warmup_start = float(ema_decay_warmup_start)
        self.ema_decay_warmup_epochs = int(ema_decay_warmup_epochs)
        self.ema_decay_warmup_curve = str(ema_decay_warmup_curve).lower()
        if self.ema_decay_warmup_curve not in ("cosine", "linear"):
            raise ValueError(f"ema_decay_warmup_curve must be 'cosine' or 'linear', got {self.ema_decay_warmup_curve!r}")
        self.ssod_start_epoch: Optional[int] = None

        # T2 MixPL (MixUp on unlabeled strong view).
        self.mixup_prob = float(mixup_prob)
        self.mixup_alpha = float(mixup_alpha)

        # T8 MIC (Tier 2 Phase 2): Masked Image Consistency on the student
        # strong view only. The default fill value uses zeros, which equals
        # ImageNet-normalized mean for already-normalized batches (mean was
        # subtracted, so zero == "mean grey"). Configurable for safety.
        self.mic_prob = float(mic_prob)
        self.mic_patch_size = int(mic_patch_size)
        self.mic_mask_ratio = float(mic_mask_ratio)
        # Default fill: zeros (matches ImageNet-normalized mean since the
        # batch has been mean-subtracted). Configurable per-channel if needed.
        if mic_fill_value is None:
            self.mic_fill_value: Sequence[float] = (0.0, 0.0, 0.0)
        else:
            self.mic_fill_value = tuple(float(v) for v in mic_fill_value)

        # T9 (Tier 2 Phase 3): pseudo-DN injection with strict guards.
        # The legacy behaviour ran DN on pseudo labels with the SAME noise
        # (label=0.5, box=1.0) used for clean GT, compounding pseudo-label
        # noise with heavy DN noise. T9 reduces both, optionally filters
        # very low-confidence pseudo labels out of the DN supervision, and
        # exposes a kill-switch ``enabled`` for ablations.
        # Literature warning (Sparse Semi-DETR, MIC paper appendix): plain
        # pseudo-DN can hurt when teacher confidence is very low; current
        # teacher has p95 ~0.05-0.08, so we keep the gate permissive by
        # default and rely mostly on reduced-noise + the adaptive threshold
        # tracker's class-level confidence filter.
        pdn_cfg = dict(pseudo_dn_cfg or {})
        self.pseudo_dn_enabled = bool(pdn_cfg.get("enabled", False))
        self.pseudo_dn_label_noise = float(pdn_cfg.get("label_noise_ratio", 0.1))
        self.pseudo_dn_box_noise = float(pdn_cfg.get("box_noise_scale", 0.4))
        self.pseudo_dn_min_score = float(pdn_cfg.get("min_score", 0.0))

        # =========================================================
        # REDESIGN v1: frozen-DINOv3 dense feature distillation.
        # The detector distills DINOv3 into a finetuned 192-d ViT-Tiny and
        # discards the rich frozen features. We re-introduce them as a
        # label-free, EMA-independent regularization target on the UNLABELED
        # stream: align the student encoder's stride-16 40x40 map to a FROZEN
        # DINOv3 ViT-S/16 dense map. This cannot participate in the EMA
        # self-training loop, so it injects information the teacher's confident
        # (easy) pseudo-labels never carry -- the diagnosed SSOD ceiling.
        # Train-only => zero inference cost.
        dcfg = dict(distill_cfg or {})
        self.distill_enabled = bool(dcfg.get("enabled", False))
        # which feature level to align: 0=stride8(80x80), 1=stride16(40x40), 2=stride32(20x20)
        self.distill_level = int(dcfg.get("level", 1))
        self.distill_loss_type = str(dcfg.get("loss_type", "cosine")).lower()
        # run distill also on MIC-masked steps (masked student -> clean target,
        # a free masked-feature-distillation signal). MixUp steps are always
        # skipped (blended image has no clean DINOv3 target).
        self.distill_on_mic = bool(dcfg.get("on_mic", True))
        self.distill_teacher: Optional[nn.Module] = None
        self.distill_proj: Optional[nn.Module] = None
        if self.distill_enabled:
            self.distill_teacher = DINOv3Backbone(
                weights_dir=dcfg.get("weights_dir", "./weights/pretrained/dinov3_vits16"),
                freeze=True,
                # SSOD batches are already ImageNet-normalized; avoid double-norm.
                normalize=bool(dcfg.get("normalize", False)),
            )
            self.distill_teacher.eval()
            for p in self.distill_teacher.parameters():
                p.requires_grad_(False)
            target_dim = int(self.distill_teacher.out_channels[self.distill_level])
            student_dim = int(dcfg.get("student_dim", 192))
            self.distill_proj = nn.Conv2d(student_dim, target_dim, kernel_size=1)
            print(
                f"REDESIGN distill ON: level={self.distill_level} "
                f"student_dim={student_dim}->target_dim={target_dim} "
                f"loss={self.distill_loss_type} weight={self.loss_weights.get('distill')} "
                f"on_mic={self.distill_on_mic}"
            )

        # =========================================================
        # Frozen-DINOv3 verifier for low-confidence pseudo-label mining. It crops
        # each teacher candidate, runs a frozen DINOv3-S CLS encoder plus a linear
        # object/background head, and uses the verifier confidence to gate or weight
        # pseudo labels independently of the detector's own confidence.
        vcfg = dict(verifier_cfg or {})
        self.verifier_enabled = bool(vcfg.get("enabled", False))
        self.verif_band_lo = float(vcfg.get("band_lo", 0.05))
        self.verif_band_hi = float(vcfg.get("band_hi", 0.40))
        self.verif_thr = float(vcfg.get("objectness_thr", 0.5))
        self.verif_max_per_img = int(vcfg.get("max_per_img", 20))
        self.verif_weight_scale = float(vcfg.get("weight_scale", 1.0))
        self.verif_expand = float(vcfg.get("expand", 0.15))
        # class-aware: only recover these class ids (rare classes). Empty/None = all.
        _vc = vcfg.get("classes", None)
        self.verif_classes = set(int(c) for c in _vc) if _vc else None
        # mode: "augment" = add band candidates; "filter" = gate and reweight mined
        # pseudo labels by verifier confidence. The filter mode lowers the detector
        # threshold to release candidates, then lets DINOv3 verify likely objects.
        self.verif_mode = str(vcfg.get("mode", "augment")).lower()
        self.verif_min_weight = float(vcfg.get("min_weight", 0.0))
        # how to weight verifier-kept boxes:
        #   "mul"  = original score-based pseudo weight * verifier confidence
        #   "objectness" = verifier confidence alone
        #   "max"  = max(original score-based weight, verifier confidence)
        self.verif_weight_mode = str(vcfg.get("weight_mode", "mul")).lower()
        self._verif_kept_last = 0
        self._verif_dropped_last = 0
        self.verifier_model = None
        self.verifier_head = None
        self._verif_added_last = 0
        if self.verifier_enabled:
            from transformers import AutoModel
            vdir = vcfg.get("weights_dir", "./weights/pretrained/dinov3_vits16")
            self.verifier_model = AutoModel.from_pretrained(vdir, local_files_only=True).eval()
            for p in self.verifier_model.parameters():
                p.requires_grad_(False)
            head_path = vcfg.get("head_path", "./weights/pretrained/verifier_head.pt")
            ck = torch.load(head_path, map_location="cpu")
            in_dim = int(ck.get("in_dim", 384))
            self.verifier_head = nn.Linear(in_dim, 1)
            self.verifier_head.load_state_dict({"weight": ck["weight"], "bias": ck["bias"]})
            self.verifier_head.eval()
            for p in self.verifier_head.parameters():
                p.requires_grad_(False)
            print(
                f"DINOv3 verifier ON: band=[{self.verif_band_lo},{self.verif_band_hi}) "
                f"objectness_thr={self.verif_thr} max/img={self.verif_max_per_img} "
                f"weight_scale={self.verif_weight_scale} head={head_path}"
            )

        # =========================================================
        # Frozen-DINOv3 multi-class head. The teacher may localize well but mislabel
        # rare classes in the medium-confidence band.
        # When this head is present, the filter verifier REASSIGNS each kept box's
        # class to the DINOv3 4-class call (the teacher supplies WHERE, DINOv3
        # supplies WHAT) and gates by a per-class accept threshold calibrated for
        # the pool. A per-class cap can throttle dominant classes so they do not
        # overwhelm rare-class gradients.
        # The binary head above still rejects background first.
        self.verifier_head4 = None
        self.verif4_accept = {}
        self.verif4_reassign = bool(vcfg.get("reassign_label", True))
        # Per-class cap on verifier-kept boxes per image (0 = no cap).
        _cap = vcfg.get("per_class_cap", None)
        self.verif4_per_class_cap = {int(k): int(v) for k, v in _cap.items()} if _cap else {}
        # Optional per-class loss scaling for long-tailed datasets. Multiplies the
        # per-box loss weight; default 1.0 means no change.
        _cls = vcfg.get("class_loss_scale", None)
        self.verif4_class_loss_scale = {int(k): float(v) for k, v in _cls.items()} if _cls else {}
        head4_path = vcfg.get("head4_path", None)
        if self.verifier_enabled and head4_path:
            ck4 = torch.load(head4_path, map_location="cpu")
            in_dim4 = int(ck4.get("in_dim", 384))
            ncls4 = int(ck4.get("num_classes", 4))
            self.verifier_head4 = nn.Linear(in_dim4, ncls4)
            self.verifier_head4.load_state_dict({"weight": ck4["weight"], "bias": ck4["bias"]})
            self.verifier_head4.eval()
            for p in self.verifier_head4.parameters():
                p.requires_grad_(False)
            # accept thresholds: head's calibrated defaults, overridable by config.
            self.verif4_accept = {int(k): float(v) for k, v in ck4.get("accept_threshold", {}).items()}
            for k, v in (vcfg.get("accept_threshold") or {}).items():
                self.verif4_accept[int(k)] = float(v)
            print(
                f"REDESIGN 4-class verifier ON: accept_thr={self.verif4_accept} "
                f"reassign={self.verif4_reassign} per_class_cap={self.verif4_per_class_cap} head4={head4_path}"
            )

        # REDESIGN v4: soft-background / ignore regions. Build, per unlabeled image,
        # an "ignore" set of probable-object regions = teacher detections with
        # score >= score_lo that were NOT promoted to positives. The criterion then
        # does not penalise unmatched student queries overlapping these as hard
        # background -> fixes the measured 33% false-negative suppression.
        sbcfg = dict(soft_bg_cfg or {})
        self.soft_bg_enabled = bool(sbcfg.get("enabled", False))
        self.soft_bg_score_lo = float(sbcfg.get("score_lo", 0.10))
        self.soft_bg_max_regions = int(sbcfg.get("max_regions", 100))
        self._soft_bg_count_last = 0
        if self.soft_bg_enabled:
            print(
                f"REDESIGN soft-background ON: score_lo={self.soft_bg_score_lo} "
                f"max_regions={self.soft_bg_max_regions} (criterion ignore_iou/weight set in DEIMCriterion cfg)"
            )

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        print(
            f"SSOD teacher EMA active: target_decay={self.ema_decay_target} "
            f"warmup_start={self.ema_decay_warmup_start} warmup_epochs={self.ema_decay_warmup_epochs} "
            f"mixup_prob={self.mixup_prob} adaptive_threshold={self.adaptive_threshold is not None}"
        )

    def to(self, device: torch.device):
        self.teacher.to(device)
        if self.distill_teacher is not None:
            self.distill_teacher.to(device)
        if self.distill_proj is not None:
            self.distill_proj.to(device)
        if self.verifier_model is not None:
            self.verifier_model.to(device)
        if self.verifier_head is not None:
            self.verifier_head.to(device)
        if self.verifier_head4 is not None:
            self.verifier_head4.to(device)
        return self

    @staticmethod
    def _cxcywh_iou(a, b):
        """IoU between one cxcywh-normalized box a (4,) and rows of b (N,4)."""
        ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
        ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
        bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
        bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
        ix1 = torch.maximum(ax1, bx1); iy1 = torch.maximum(ay1, by1)
        ix2 = torch.minimum(ax2, bx2); iy2 = torch.minimum(ay2, by2)
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        ua = a[2] * a[3] + b[:, 2] * b[:, 3] - inter
        return inter / ua.clamp(min=1e-6)

    @torch.no_grad()
    def _verifier_augment(self, high_targets_weak, teacher_outputs, weak_images, autocast_kwargs):
        """Recover under-confident objects from low-confidence teacher candidates
        using the frozen DINOv3 object/background verifier."""
        from torchvision.ops import roi_align
        B = weak_images.shape[0]
        Hh, Ww = weak_images.shape[-2], weak_images.shape[-1]
        out = []
        added = 0
        for i in range(B):
            tgt = high_targets_weak[i]
            scores, labels, boxes = _prediction_view(teacher_outputs, i)  # boxes cxcywh norm [0,1]
            band = (scores >= self.verif_band_lo) & (scores < self.verif_band_hi)
            if self.verif_classes:
                cls_mask = torch.zeros_like(band)
                for c in self.verif_classes:
                    cls_mask |= (labels == c)
                band = band & cls_mask
            idx = torch.nonzero(band, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                out.append(tgt); continue
            cb = boxes[idx]; cs = scores[idx]; cl = labels[idx]
            cx, cy, bw, bh = cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3]
            ex = bw * (1.0 + 2 * self.verif_expand); ey = bh * (1.0 + 2 * self.verif_expand)
            x1 = (cx - ex / 2).clamp(0, 1) * Ww; y1 = (cy - ey / 2).clamp(0, 1) * Hh
            x2 = (cx + ex / 2).clamp(0, 1) * Ww; y2 = (cy + ey / 2).clamp(0, 1) * Hh
            valid = (x2 > x1 + 1) & (y2 > y1 + 1)
            if valid.sum() == 0:
                out.append(tgt); continue
            vi = torch.nonzero(valid, as_tuple=False).squeeze(1)
            rois = torch.stack([torch.zeros_like(x1[vi]), x1[vi], y1[vi], x2[vi], y2[vi]], dim=1)
            with torch.autocast(**autocast_kwargs):
                crops = roi_align(weak_images[i:i + 1], rois, output_size=(224, 224), spatial_scale=1.0, aligned=True)
                cls = self.verifier_model(pixel_values=crops).last_hidden_state[:, 0, :].float()
            cls = F.normalize(cls, dim=1)
            objectness = torch.sigmoid(self.verifier_head(cls).squeeze(1))
            acc = objectness >= self.verif_thr
            if acc.sum() == 0:
                out.append(tgt); continue
            sel = vi[acc]; osel = objectness[acc]
            if sel.numel() > self.verif_max_per_img:
                top = torch.topk(osel, self.verif_max_per_img).indices
                sel = sel[top]; osel = osel[top]
            add_boxes = cb[sel]; add_labels = cl[sel].long(); add_scores = cs[sel]
            add_weights = osel * self.verif_weight_scale
            # drop candidates that duplicate an existing high-conf pseudo box
            if tgt["boxes"].numel() > 0:
                keep = []
                for k in range(add_boxes.shape[0]):
                    if self._cxcywh_iou(add_boxes[k], tgt["boxes"]).max() < 0.5:
                        keep.append(k)
                if len(keep) < add_boxes.shape[0]:
                    keep = torch.tensor(keep, device=add_boxes.device, dtype=torch.long)
                    add_boxes = add_boxes[keep]; add_labels = add_labels[keep]
                    add_scores = add_scores[keep]; add_weights = add_weights[keep]
            if add_boxes.shape[0] == 0:
                out.append(tgt); continue
            out.append({
                "boxes": torch.cat([tgt["boxes"], add_boxes], 0),
                "labels": torch.cat([tgt["labels"], add_labels], 0),
                "scores": torch.cat([tgt["scores"], add_scores], 0),
                "weights": torch.cat([tgt["weights"], add_weights], 0),
            })
            added += int(add_boxes.shape[0])
        self._verif_added_last = added
        return out

    @torch.no_grad()
    def _verifier_filter(self, targets_weak, weak_images, autocast_kwargs):
        """Score each released pseudo-label box with the frozen DINOv3 verifier,
        drop boxes below the objectness threshold, and soft-reweight survivors.
        If verif_classes is set, only those classes are gated."""
        from torchvision.ops import roi_align
        Hh, Ww = weak_images.shape[-2], weak_images.shape[-1]
        rois = []; meta = []  # meta entries: (img_i, local_j, valid)
        for i, tgt in enumerate(targets_weak):
            b = tgt["boxes"]
            for j in range(int(b.shape[0])):
                cx, cy, bw, bh = b[j]
                ex = bw * (1.0 + 2 * self.verif_expand); ey = bh * (1.0 + 2 * self.verif_expand)
                x1 = float((cx - ex / 2).clamp(0, 1) * Ww); y1 = float((cy - ey / 2).clamp(0, 1) * Hh)
                x2 = float((cx + ex / 2).clamp(0, 1) * Ww); y2 = float((cy + ey / 2).clamp(0, 1) * Hh)
                if x2 > x1 + 1 and y2 > y1 + 1:
                    rois.append([i, x1, y1, x2, y2]); meta.append((i, j, True))
                else:
                    meta.append((i, j, False))
        objectness = None; probs4 = None
        if rois:
            rt = torch.tensor(rois, device=weak_images.device, dtype=weak_images.dtype)
            ochunks = []; pchunks = []
            for s in range(0, rt.shape[0], 128):
                with torch.autocast(**autocast_kwargs):
                    crops = roi_align(weak_images, rt[s:s + 128], output_size=(224, 224), spatial_scale=1.0, aligned=True)
                    cls = self.verifier_model(pixel_values=crops).last_hidden_state[:, 0, :].float()
                cls = F.normalize(cls, dim=1)
                ochunks.append(torch.sigmoid(self.verifier_head(cls).squeeze(1)))
                if self.verifier_head4 is not None:
                    pchunks.append(torch.softmax(self.verifier_head4(cls), dim=1))
            objectness = torch.cat(ochunks)
            if pchunks:
                probs4 = torch.cat(pchunks)
        obj_per = {}; cls_per = {}; conf_per = {}
        bidx = 0
        for (i, j, ok) in meta:
            if ok:
                obj_per[(i, j)] = float(objectness[bidx])
                if probs4 is not None:
                    p = probs4[bidx]; c = int(p.argmax())
                    cls_per[(i, j)] = c; conf_per[(i, j)] = float(p[c])
                bidx += 1
            else:
                obj_per[(i, j)] = 0.0; cls_per[(i, j)] = -1; conf_per[(i, j)] = 0.0
        use4 = self.verifier_head4 is not None
        out = []; kept = 0; dropped = 0
        for i, tgt in enumerate(targets_weak):
            n = int(tgt["boxes"].shape[0])
            if n == 0:
                out.append(tgt); continue
            dev = tgt["boxes"].device
            obj = torch.tensor([obj_per[(i, j)] for j in range(n)], device=dev)
            object_gate = obj >= self.verif_thr

            if not use4:
                # Legacy binary behavior: gate and soft-reweight survivors.
                if self.verif_classes:
                    lab = tgt["labels"]
                    inscope = torch.zeros(n, dtype=torch.bool, device=lab.device)
                    for c in self.verif_classes:
                        inscope |= (lab == c)
                    keep = object_gate | (~inscope)
                else:
                    keep = object_gate
                ki = torch.nonzero(keep, as_tuple=False).squeeze(1)
                obj_c = obj.clamp(min=self.verif_min_weight, max=1.0)
                if self.verif_weight_mode == "objectness":
                    w = obj_c
                elif self.verif_weight_mode == "max":
                    w = torch.maximum(tgt["weights"], obj_c)
                else:  # "mul"
                    w = tgt["weights"].clone() * obj_c
                out.append({
                    "boxes": tgt["boxes"][ki], "labels": tgt["labels"][ki],
                    "scores": tgt["scores"][ki], "weights": w[ki],
                })
                kept += int(ki.numel()); dropped += int((~keep).sum())
                continue

            # 4-class path: teacher supplies WHERE, DINOv3 supplies WHAT.
            c4 = torch.tensor([cls_per[(i, j)] for j in range(n)], device=dev, dtype=torch.long)
            p4 = torch.tensor([conf_per[(i, j)] for j in range(n)], device=dev)
            accept = torch.zeros(n, dtype=torch.bool, device=dev)
            for j in range(n):
                cj = int(c4[j])
                if cj >= 0 and float(p4[j]) >= self.verif4_accept.get(cj, 0.5):
                    accept[j] = True
            keep = object_gate & accept
            ki = torch.nonzero(keep, as_tuple=False).squeeze(1)
            new_labels = c4 if self.verif4_reassign else tgt["labels"]
            # Per-class cap for dominant classes: keep top-confidence boxes only.
            if self.verif4_per_class_cap and ki.numel() > 0:
                kl = new_labels[ki]; kp = p4[ki]
                keep_mask = torch.ones(ki.numel(), dtype=torch.bool, device=dev)
                for cj, cap in self.verif4_per_class_cap.items():
                    pos = torch.nonzero(kl == int(cj), as_tuple=False).squeeze(1)
                    if pos.numel() > int(cap):
                        drop = pos[torch.argsort(kp[pos], descending=True)[int(cap):]]
                        keep_mask[drop] = False
                ki = ki[keep_mask]
            # loss weight = DINOv3 class confidence p* x per-class scale (criterion
            # uses scores*weights). scores=p* (quality), weights=class scale, so the
            # effective gradient weight is p* * class_scale (rare classes up-weighted).
            if self.verif4_class_loss_scale and ki.numel() > 0:
                cs = torch.tensor(
                    [self.verif4_class_loss_scale.get(int(l), 1.0) for l in new_labels[ki].tolist()],
                    device=dev, dtype=p4.dtype,
                )
            else:
                cs = torch.ones(ki.numel(), device=dev, dtype=p4.dtype)
            out.append({
                "boxes": tgt["boxes"][ki],
                "labels": new_labels[ki],
                "scores": p4[ki],
                "weights": cs,
            })
            kept += int(ki.numel()); dropped += int(n - ki.numel())
        self._verif_kept_last = kept; self._verif_dropped_last = dropped
        return out

    @torch.no_grad()
    def _build_ignore_targets(self, teacher_outputs):
        """Per image, probable-object regions (teacher detections with score
        >= soft_bg_score_lo). Returned as weak-view target dicts so they can be
        mapped to the strong view alongside the pseudo-labels. Capped by score."""
        B = teacher_outputs["pred_logits"].shape[0]
        out = []
        for i in range(B):
            scores, labels, boxes = _prediction_view(teacher_outputs, i)
            idx = torch.nonzero(scores >= self.soft_bg_score_lo, as_tuple=False).squeeze(1)
            if idx.numel() > self.soft_bg_max_regions:
                top = torch.topk(scores[idx], self.soft_bg_max_regions).indices
                idx = idx[top]
            out.append({"boxes": boxes[idx], "labels": labels[idx].long()})
        return out

    def _current_ema_decay(self) -> float:
        """Compute the EMA decay to apply this step.

        Three regimes:
        1. Burn-in (ssod_start_epoch is None): use ``ema_decay_burnin`` if
           configured. Default behaviour (None) keeps the legacy target decay
           so existing configs don't shift unexpectedly.
        2. SSOD warmup window: ramp from ``ema_decay_warmup_start`` toward
           ``ema_decay_target`` over ``ema_decay_warmup_epochs`` epochs. A
           cosine curve (default) avoids the hard discontinuity at the
           burn-in -> SSOD boundary that linearly ramping produces. This is
           the Mean Teacher convention.
        3. Steady-state SSOD: use ``ema_decay_target``.
        """
        if self.ssod_start_epoch is None or self.current_epoch is None:
            return self.ema_decay_burnin if self.ema_decay_burnin is not None else self.ema_decay_target
        if self.ema_decay_warmup_epochs <= 0:
            return self.ema_decay_target
        elapsed = float(self.current_epoch) - float(self.ssod_start_epoch)
        if elapsed <= 0.0:
            return self.ema_decay_warmup_start
        if elapsed >= self.ema_decay_warmup_epochs:
            return self.ema_decay_target
        frac = elapsed / float(self.ema_decay_warmup_epochs)
        if self.ema_decay_warmup_curve == "cosine":
            # Standard half-cosine ramp: smoothly rises from start to target.
            import math
            ramp = 0.5 * (1.0 - math.cos(math.pi * frac))
        else:
            ramp = frac
        return self.ema_decay_warmup_start + ramp * (self.ema_decay_target - self.ema_decay_warmup_start)

    @torch.no_grad()
    def update_teacher_ema(self):
        student = dist_utils.de_parallel(self.student)
        teacher = dist_utils.de_parallel(self.teacher)
        student_params = dict(student.named_parameters())
        ema_delta = None
        decay = float(self._current_ema_decay())
        # Keep ``self.ema_decay`` reflecting the value actually applied, so the
        # solver's diagnostic logging stays informative.
        self.ema_decay = decay

        # Defensive: if AMP/optimizer produced any NaN/Inf in the student
        # weights, copying them into the teacher poisons every future step.
        # Skip the EMA update on that step; the next step will retry once the
        # GradScaler has lowered the loss scale and the student is finite
        # again. Without this guard the trainer raised
        # "teacher.pred_logits contains NaN" mid-run during stress tests with
        # tiny batch + zero burn-in.
        student_finite = True
        for pname, sparam in student_params.items():
            if not sparam.dtype.is_floating_point:
                continue
            if not torch.isfinite(sparam).all():
                student_finite = False
                self._ema_skip_count = getattr(self, "_ema_skip_count", 0) + 1
                if self._ema_skip_count <= 5 or self._ema_skip_count % 50 == 0:
                    print(
                        f"[SSOD] Skipping teacher EMA update at epoch {self.current_epoch}: "
                        f"non-finite student parameter '{pname}' (skip_count={self._ema_skip_count})"
                    )
                break

        if student_finite:
            for name, teacher_param in teacher.named_parameters():
                student_param = student_params[name].detach()
                if not teacher_param.dtype.is_floating_point:
                    teacher_param.copy_(student_param)
                    continue
                old_value = teacher_param.detach().clone() if ema_delta is None and student_params[name].requires_grad else None
                teacher_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
                if old_value is not None:
                    ema_delta = (teacher_param - old_value).abs().mean()

            student_buffers = dict(student.named_buffers())
            for name, teacher_buffer in teacher.named_buffers():
                if name in student_buffers:
                    teacher_buffer.copy_(student_buffers[name])

        if ema_delta is None:
            device = next(teacher.parameters()).device
            ema_delta = torch.zeros((), device=device)
        return ema_delta.detach()

    def _criterion_loss(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device_type = outputs["pred_logits"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            loss_dict = self.criterion(outputs, targets, epoch=self.current_epoch)
        return sum(loss_dict.values()), loss_dict

    def _weighted_prefixed(self, prefix: str, loss_dict: Dict[str, torch.Tensor], weight: float) -> Dict[str, torch.Tensor]:
        return {f"{prefix}_{name}": value.detach() * weight for name, value in loss_dict.items()}

    @torch.no_grad()
    def _sample_box_features(
        self,
        feat: torch.Tensor,
        boxes_per_sample: List[torch.Tensor],
        query_count: int,
        pool_size: int,
    ) -> torch.Tensor:
        """ROIAlign-lite: sample ``pool_size x pool_size`` features per box and
        mean-pool. ``boxes_per_sample[i]`` must be in the coordinate system of
        ``feat[i]`` (normalized cxcywh on the same canvas).

        Returns a tensor of shape (batch, query_count, channels). Slots beyond
        a sample's actual box count are left at zero; the caller is expected
        to mask these out via ``weights``.
        """
        batch_size, channels, _, _ = feat.shape
        content = feat.new_zeros((batch_size, query_count, channels))
        if query_count <= 0:
            return content
        offsets = torch.linspace(
            0.5 / pool_size,
            1.0 - 0.5 / pool_size,
            pool_size,
            device=feat.device,
            dtype=feat.dtype,
        )
        for batch_idx, target_boxes in enumerate(boxes_per_sample):
            if target_boxes.numel() == 0:
                continue
            boxes = target_boxes.to(device=feat.device, dtype=feat.dtype).clamp(0.0, 1.0)
            num = min(boxes.shape[0], query_count)
            boxes = boxes[:num]
            xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
            x0, y0, x1, y1 = xyxy.unbind(-1)
            xs = x0[:, None] * (1.0 - offsets[None, :]) + x1[:, None] * offsets[None, :]
            ys = y0[:, None] * (1.0 - offsets[None, :]) + y1[:, None] * offsets[None, :]
            grid_x = xs[:, None, :].expand(-1, pool_size, -1)
            grid_y = ys[:, :, None].expand(-1, -1, pool_size)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.mul(2.0).sub(1.0)
            sampled = F.grid_sample(
                feat[batch_idx:batch_idx + 1].expand(num, -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            content[batch_idx, :num] = sampled.mean(dim=(-1, -2))
        return content

    @torch.no_grad()
    def _build_paired_consistency_queries(
        self,
        teacher_encoded_feats: List[torch.Tensor],
        weak_targets: List[Dict[str, torch.Tensor]],
        strong_targets: List[Dict[str, torch.Tensor]],
        max_queries: int,
        pool_size: int,
        min_score: float,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Build a *pair* of consistency_queries dicts for cross-view CVC.

        Same content (sampled from teacher's weak feature at weak-view box
        positions) but different reference boxes:
        - teacher dict: weak-view boxes (consumed by teacher decoder)
        - student dict: strong-view boxes (consumed by student decoder)

        This corrects the prior bug where the strong-view boxes were used to
        sample weak-view teacher features, producing geometrically nonsensical
        content embeddings whenever the strong view had any crop/zoomout/flip
        (>=75% of training steps in our config).

        Queries with ``score < min_score`` get a zero weight so they contribute
        no gradient -- this is the literature-recommended "noise-amplifying"
        guard for CVC under uncertain pseudo-labels.
        """
        if not teacher_encoded_feats:
            return None
        feat = teacher_encoded_feats[0]

        # Determine per-sample weak/strong pairs. If augmentation mapping dropped
        # boxes, simple prefix pairing is wrong; use the carried pair id to keep
        # the weak teacher content aligned to the corresponding strong box.
        pair_indices: List[List[Tuple[int, int]]] = []
        for w_target, s_target in zip(weak_targets, strong_targets):
            n_w = int(w_target["boxes"].shape[0]) if w_target["boxes"].numel() else 0
            n_s = int(s_target["boxes"].shape[0]) if s_target["boxes"].numel() else 0
            if n_w <= 0 or n_s <= 0:
                pair_indices.append([])
                continue

            weak_ids = w_target.get("_pseudo_pair_id")
            strong_ids = s_target.get("_pseudo_pair_id")
            pairs: List[Tuple[int, int]] = []
            if weak_ids is not None and strong_ids is not None and weak_ids.numel() > 0 and strong_ids.numel() > 0:
                strong_lookup = {int(pid): idx for idx, pid in enumerate(strong_ids[:n_s].detach().cpu().tolist())}
                for weak_idx, pid in enumerate(weak_ids[:n_w].detach().cpu().tolist()):
                    strong_idx = strong_lookup.get(int(pid))
                    if strong_idx is not None:
                        pairs.append((weak_idx, strong_idx))
            else:
                pairs = [(idx, idx) for idx in range(min(n_w, n_s))]

            scores = w_target.get("scores")
            if scores is not None and scores.numel() > 0 and pairs:
                pairs.sort(key=lambda item: float(scores[item[0]].detach().cpu()), reverse=True)
            pair_indices.append(pairs[:max_queries])

        query_count = max((len(pairs) for pairs in pair_indices), default=0)
        if query_count <= 0:
            return None

        weak_boxes_per_sample: List[torch.Tensor] = []
        strong_boxes_per_sample: List[torch.Tensor] = []
        weights_per_sample: List[torch.Tensor] = []

        for pairs, w_target, s_target in zip(pair_indices, weak_targets, strong_targets):
            n_paired = len(pairs)
            if n_paired <= 0:
                weak_boxes_per_sample.append(feat.new_zeros((0, 4)))
                strong_boxes_per_sample.append(feat.new_zeros((0, 4)))
                weights_per_sample.append(feat.new_zeros((0,)))
                continue
            weak_idx = torch.as_tensor([item[0] for item in pairs], device=w_target["boxes"].device, dtype=torch.long)
            strong_idx = torch.as_tensor([item[1] for item in pairs], device=s_target["boxes"].device, dtype=torch.long)

            # Ranked by teacher score above; now gather the exact paired entries.
            scores = w_target.get("scores")
            if scores is None or scores.numel() == 0:
                scores = torch.ones(n_paired, device=feat.device, dtype=feat.dtype)
            else:
                scores = scores[weak_idx].to(device=feat.device, dtype=feat.dtype)
            weights = w_target.get("weights")
            if weights is None or weights.numel() == 0:
                weights = torch.ones_like(scores)
            else:
                weights = weights[weak_idx].to(device=feat.device, dtype=feat.dtype)
            # Confidence gate: zero out weights below ``min_score``. Decoder
            # still processes these slots (we need consistent shape) but the
            # consistency loss multiplies by ``weights`` so they contribute
            # nothing to the gradient.
            if min_score > 0:
                weights = weights * (scores >= float(min_score)).to(weights.dtype)

            w_boxes = w_target["boxes"][weak_idx].to(device=feat.device, dtype=feat.dtype).clamp(0.0, 1.0)
            s_boxes = s_target["boxes"][strong_idx].to(device=feat.device, dtype=feat.dtype).clamp(0.0, 1.0)
            weak_boxes_per_sample.append(w_boxes)
            strong_boxes_per_sample.append(s_boxes)
            weights_per_sample.append(weights)

        # Sample content from teacher weak feature at WEAK-view box coordinates.
        content = self._sample_box_features(
            feat, weak_boxes_per_sample, query_count=query_count, pool_size=pool_size
        )

        # Pad boxes / weights to (B, query_count, ...).
        batch_size = feat.shape[0]
        weak_boxes_out = feat.new_full((batch_size, query_count, 4), 0.5)
        strong_boxes_out = feat.new_full((batch_size, query_count, 4), 0.5)
        weights_out = feat.new_zeros((batch_size, query_count))
        for batch_idx, (w, s, wt) in enumerate(zip(weak_boxes_per_sample, strong_boxes_per_sample, weights_per_sample)):
            num = w.shape[0]
            if num == 0:
                continue
            weak_boxes_out[batch_idx, :num] = w
            strong_boxes_out[batch_idx, :num] = s
            weights_out[batch_idx, :num] = wt

        if not (weights_out > 0).any():
            return None

        teacher_queries = {
            "content": content.detach(),
            "boxes": weak_boxes_out,
            "weights": weights_out,
        }
        student_queries = {
            "content": content.detach(),
            "boxes": strong_boxes_out,
            "weights": weights_out,
        }
        return teacher_queries, student_queries

    @staticmethod
    def _attach_pseudo_pair_ids(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Attach stable per-sample pseudo ids before weak->strong mapping."""
        with_ids: List[Dict[str, torch.Tensor]] = []
        for target in targets:
            new_target = dict(target)
            n = int(target["labels"].numel())
            new_target["_pseudo_pair_id"] = torch.arange(n, device=target["labels"].device, dtype=torch.long)
            with_ids.append(new_target)
        return with_ids

    def _maybe_mixup_strong(
        self,
        strong_images: torch.Tensor,
        high_targets: List[Dict[str, torch.Tensor]],
        enable_unsupervised: bool,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], Dict[str, float]]:
        """MixPL-style MixUp on the unlabeled strong view.

        Picks a random in-batch permutation, linearly blends the strong images
        and concatenates the corresponding pseudo-targets. Coordinates are
        already normalized cxcywh on the same canvas, so concatenation alone
        is geometrically correct. The target weights of the second source are
        scaled by ``1 - alpha`` so the criterion treats them with reduced
        confidence -- mirroring the soft-label semantics from MixPL.
        """
        stats = {
            "mixup_applied_count": 0.0,
            "mixup_alpha_mean": 0.0,
            "mixup_added_labels": 0.0,
        }
        if not enable_unsupervised or self.mixup_prob <= 0.0 or strong_images.shape[0] < 2:
            return strong_images, high_targets, stats
        if random.random() >= self.mixup_prob:
            return strong_images, high_targets, stats
        if self.mixup_alpha > 0:
            alpha = float(
                torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
                .sample()
                .item()
            )
        else:
            alpha = 0.5
        alpha = float(min(max(alpha, 0.05), 0.95))
        batch_size = strong_images.shape[0]
        perm = torch.randperm(batch_size, device=strong_images.device)
        mixed_images = alpha * strong_images + (1.0 - alpha) * strong_images[perm]
        mixed_targets: List[Dict[str, torch.Tensor]] = []
        added = 0
        for batch_idx in range(batch_size):
            primary = high_targets[batch_idx]
            secondary = high_targets[int(perm[batch_idx].item())]
            if secondary["labels"].numel() == 0:
                mixed_targets.append(primary)
                continue
            secondary_weights = secondary.get("weights")
            if secondary_weights is None:
                secondary_weights = torch.ones_like(secondary["scores"])
            scaled_secondary_weights = secondary_weights * (1.0 - alpha)
            primary_weights = primary.get("weights")
            if primary_weights is None:
                primary_weights = torch.ones_like(primary["scores"])
            scaled_primary_weights = primary_weights * alpha
            mixed_targets.append({
                "boxes": torch.cat([primary["boxes"], secondary["boxes"]], dim=0),
                "labels": torch.cat([primary["labels"], secondary["labels"]], dim=0),
                "scores": torch.cat([primary["scores"], secondary["scores"]], dim=0),
                "weights": torch.cat([scaled_primary_weights, scaled_secondary_weights], dim=0),
            })
            added += int(secondary["labels"].numel())
        stats["mixup_applied_count"] = 1.0
        stats["mixup_alpha_mean"] = float(alpha)
        stats["mixup_added_labels"] = float(added)
        return mixed_images, mixed_targets, stats

    def _pseudo_dn_override(self) -> Optional[Dict[str, float]]:
        """Build a per-call DN noise override dict for SSOD branches.

        Returns ``None`` when pseudo-DN is disabled, so the decoder uses its
        default (clean-GT) noise scales -- this is the safe fallback for the
        labeled supervised branch.
        """
        if not self.pseudo_dn_enabled:
            return None
        return {
            "label_noise_ratio": float(self.pseudo_dn_label_noise),
            "box_noise_scale": float(self.pseudo_dn_box_noise),
        }

    @staticmethod
    def _filter_targets_by_score(
        targets: List[Dict[str, torch.Tensor]], min_score: float
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
        """Drop pseudo targets whose ``scores`` fall below ``min_score``.

        Returns (filtered_targets, kept_count, total_count). Targets without a
        ``scores`` field are passed through unchanged.
        """
        if min_score <= 0.0:
            total = sum(int(t["labels"].numel()) for t in targets)
            return targets, total, total
        filtered: List[Dict[str, torch.Tensor]] = []
        kept = 0
        total = 0
        for target in targets:
            scores = target.get("scores")
            n = int(target["labels"].numel())
            total += n
            if scores is None or scores.numel() == 0:
                filtered.append(target)
                kept += n
                continue
            mask = scores >= float(min_score)
            new_target: Dict[str, torch.Tensor] = {}
            for key, value in target.items():
                # ``ignore_boxes`` are not per-label entries -- never index them by the
                # per-label score mask even if their count coincides with n.
                if key != "ignore_boxes" and isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n:
                    new_target[key] = value[mask]
                else:
                    new_target[key] = value
            filtered.append(new_target)
            kept += int(new_target["labels"].numel())
        return filtered, kept, total

    def train_step(
        self,
        labeled_batch,
        unlabeled_batch,
        synthetic_batch,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler=None,
        max_norm: float = 0.0,
        enable_unsupervised: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self.student.train()
        self.criterion.train()
        self.teacher.eval()

        labeled_images, labeled_targets = _move_batch(labeled_batch, device)
        unlabeled_images, _ = _move_batch(unlabeled_batch, device)
        if synthetic_batch is not None and self.loss_weights.get("synthetic", 0.0) > 0:
            synthetic_images, synthetic_targets = _move_batch(synthetic_batch, device)
        else:
            synthetic_images, synthetic_targets = None, None

        pair_aug = self.augmentor(unlabeled_images)
        weak_images = pair_aug.weak_images
        strong_images = pair_aug.strong_images
        amp_enabled = scaler is not None
        autocast_kwargs = {"device_type": device.type, "enabled": amp_enabled, "cache_enabled": True}

        with torch.no_grad():
            teacher_encoded_feats = None
            with torch.autocast(**autocast_kwargs):
                if hasattr(self.teacher, "extract_encoded_features") and hasattr(self.teacher, "decode_features"):
                    teacher_encoded_feats = self.teacher.extract_encoded_features(weak_images)
                    teacher_outputs = self.teacher.decode_features(teacher_encoded_feats)
                else:
                    teacher_outputs = self.teacher(weak_images)
            _assert_finite_outputs(teacher_outputs, "teacher")
            # T1: refresh the per-class adaptive threshold tracker BEFORE mining.
            # Only update once unsupervised training is actually active to avoid
            # polluting the rolling history during burn-in (when the teacher is
            # still cold and would bias the threshold downward).
            class_thresholds = None
            if self.adaptive_threshold is not None:
                if enable_unsupervised:
                    self.adaptive_threshold.update(teacher_outputs["pred_logits"].detach())
                class_thresholds = self.adaptive_threshold.get_thresholds()
            mining_extra = {}
            if self.num_classes is not None:
                mining_extra["num_classes"] = int(self.num_classes)
            if class_thresholds is not None:
                mining_extra["class_thresholds"] = class_thresholds
                # A. mining revival: derive a looser per-class threshold set
                # for the candidate branch. Without this, candidate == high
                # and mined was effectively empty (fix6 acceptance 0.02%).
                # Floor by adaptive_threshold.min_thresh (0.05) so candidates
                # still sit in the real-detection band.
                if 0.0 < self.candidate_threshold_scale < 1.0:
                    floor = (
                        float(self.adaptive_threshold.min_thresh)
                        if self.adaptive_threshold is not None
                        else 0.05
                    )
                    cand_thr = {
                        int(c): max(floor, float(t) * float(self.candidate_threshold_scale))
                        for c, t in class_thresholds.items()
                    }
                    mining_extra["candidate_class_thresholds"] = cand_thr
            high_mining_cfg = dict(self.mining_cfg)
            high_mining_cfg["max_mined_labels"] = 0
            high_targets_weak, _, high_stats = cost_based_pseudo_mining(
                teacher_outputs,
                score_threshold=self.pseudo_threshold,
                **high_mining_cfg,
                **mining_extra,
            )
            # Recover under-confident objects from low-confidence teacher candidates
            # via the frozen-DINOv3 verifier, before pairing.
            self._verif_added_last = 0; self._verif_kept_last = 0; self._verif_dropped_last = 0
            if enable_unsupervised and self.verifier_enabled:
                if self.verif_mode == "filter":
                    high_targets_weak = self._verifier_filter(
                        high_targets_weak, weak_images, autocast_kwargs
                    )
                else:
                    high_targets_weak = self._verifier_augment(
                        high_targets_weak, teacher_outputs, weak_images, autocast_kwargs
                    )
            high_targets_weak = self._attach_pseudo_pair_ids(high_targets_weak)
            high_targets, mapping_stats = pair_aug.map_targets_to_strong(high_targets_weak)
            teacher_outputs_strong = pair_aug.map_outputs_to_strong(teacher_outputs)
            # Attach strong-view ignore regions (probable objects not promoted to
            # positives) so the criterion does not penalise them as hard
            # background. Survives _filter_targets_by_score (M != num_labels) and the
            # decoder strip (criterion sees full high_targets). MixPL must stay off
            # (its target rebuild would drop the key) -- v4 config sets mixup_prob 0.
            self._soft_bg_count_last = 0
            if enable_unsupervised and self.soft_bg_enabled:
                ignore_weak = self._build_ignore_targets(teacher_outputs)
                ignore_strong, _ = pair_aug.map_targets_to_strong(ignore_weak)
                for i in range(len(high_targets)):
                    ib = ignore_strong[i]["boxes"]
                    high_targets[i]["ignore_boxes"] = ib
                    self._soft_bg_count_last += int(ib.shape[0])

        loss_records: Dict[str, torch.Tensor] = {}
        optimizer.zero_grad(set_to_none=True)

        # T9: build pseudo-DN noise override (reduced noise) and optionally
        # filter very low-confidence pseudo labels so the DN branch only sees
        # the better half of the pseudo-label distribution. The filter applies
        # to BOTH the main pseudo loss AND DN (we don't double-process
        # student forwards). When ``min_score=0`` this is a no-op and the
        # adaptive-threshold tracker remains the sole gate.
        pseudo_dn_override = self._pseudo_dn_override()
        high_targets_pre_filter_count = _target_count(high_targets)
        high_targets, pseudo_dn_kept, pseudo_dn_total = self._filter_targets_by_score(
            high_targets, self.pseudo_dn_min_score
        )
        # high_targets_weak is also filtered to stay aligned with high_targets
        # (CVC pairs the two), but only when min_score > 0 (avoid wasted work).
        if self.pseudo_dn_min_score > 0.0:
            high_targets_weak, _, _ = self._filter_targets_by_score(
                high_targets_weak, self.pseudo_dn_min_score
            )

        # T2 MixPL: blend two unlabeled strong-view samples + their pseudo
        # targets. Must run before the student forward so the geometry stays
        # consistent across mining, CVC and loss computation.
        strong_images, high_targets, mixup_stats = self._maybe_mixup_strong(
            strong_images, high_targets, enable_unsupervised
        )

        # REDESIGN: snapshot the CLEAN strong-view image (post-MixUp, pre-MIC)
        # to use as the frozen-DINOv3 distillation target input. Cloned so the
        # in-place-safe MIC step below cannot corrupt it. On MixUp steps this is
        # a blend and distillation is skipped (see below).
        distill_input = strong_images.detach().clone() if self.distill_enabled else None

        # T8 MIC: randomly mask patches on the STUDENT input only (teacher
        # has already produced its predictions on weak_images and never sees
        # the mask). Applied AFTER MixUp so the mask sees the final student
        # image. Targets and box coordinates are not affected by masking;
        # only the image pixels.
        mic_stats = {"mic_applied_count": 0.0, "mic_mask_fraction": 0.0}
        if enable_unsupervised and self.mic_prob > 0.0 and random.random() < self.mic_prob:
            strong_images, achieved = apply_mic_mask(
                strong_images,
                patch_size=self.mic_patch_size,
                mask_ratio=self.mic_mask_ratio,
                fill_value=self.mic_fill_value,
            )
            mic_stats["mic_applied_count"] = 1.0
            mic_stats["mic_mask_fraction"] = float(achieved)

        with torch.autocast(**autocast_kwargs):
            supervised_outputs = self.student(labeled_images, targets=_strip_aux_target_fields(labeled_targets))
        _assert_finite_outputs(supervised_outputs, "student_supervised")
        loss_sup, loss_sup_dict = self._criterion_loss(supervised_outputs, labeled_targets)

        if synthetic_images is not None:
            with torch.autocast(**autocast_kwargs):
                synthetic_outputs = self.student(synthetic_images, targets=_strip_aux_target_fields(synthetic_targets))
            _assert_finite_outputs(synthetic_outputs, "student_synthetic")
            loss_syn, loss_syn_dict = self._criterion_loss(synthetic_outputs, synthetic_targets)
        else:
            loss_syn = loss_sup * 0.0
            loss_syn_dict = {}

        # REDESIGN: split the strong-view forward into encoder + decoder so the
        # encoder feature map is computed ONCE and reused for the detection
        # loss, the mined loss, CVC, and distillation. This is exactly
        # equivalent to self.student(strong_images, targets, dn_noise_override)
        # (DEIM.forward == decode_features(extract_encoded_features(x), ...))
        # but removes up to 2 redundant backbone+encoder passes per step.
        use_split_strong = hasattr(self.student, "extract_encoded_features") and hasattr(self.student, "decode_features")
        strong_encoded_feats = None
        with torch.autocast(**autocast_kwargs):
            if use_split_strong:
                strong_encoded_feats = self.student.extract_encoded_features(strong_images)
                unsup_outputs = self.student.decode_features(
                    strong_encoded_feats,
                    targets=_strip_aux_target_fields(high_targets),
                    dn_noise_override=pseudo_dn_override,
                )
            else:
                unsup_outputs = self.student(
                    strong_images,
                    targets=_strip_aux_target_fields(high_targets),
                    dn_noise_override=pseudo_dn_override,
                )
        _assert_finite_outputs(unsup_outputs, "student_unsupervised")
        high_label_count = _target_count(high_targets)
        if enable_unsupervised and high_label_count > 0:
            loss_unsup_high, loss_unsup_high_dict = self._criterion_loss(unsup_outputs, high_targets)
        else:
            loss_unsup_high = loss_sup * 0.0
            loss_unsup_high_dict = {}

        with torch.no_grad():
            mixup_was_applied = mixup_stats.get("mixup_applied_count", 0.0) > 0.0
            if mixup_was_applied:
                # ``teacher_outputs_strong`` was derived from the pre-mixup
                # images, so it cannot be coherently matched against the
                # student forward on the mixed images. Emit empty mined targets
                # for this step and let the high-confidence branch carry the
                # supervision. The mined branch is intentionally optional.
                device_ref = teacher_outputs_strong["pred_boxes"].device
                empty_count = teacher_outputs_strong["pred_boxes"].shape[0]
                mined_targets = [
                    {
                        "boxes": torch.zeros((0, 4), device=device_ref),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device_ref),
                        "scores": torch.zeros((0,), device=device_ref),
                        "weights": torch.zeros((0,), device=device_ref),
                    }
                    for _ in range(empty_count)
                ]
                mined_stats = {
                    "mined_labels": 0.0,
                    "consistency_labels": 0.0,
                    "candidate_initial_labels": 0.0,
                    "candidate_query_labels": 0.0,
                    "matched_pairs": 0.0,
                    "accepted_mined_labels": 0.0,
                    "mined_acceptance_rate": 0.0,
                    "mined_target_weight_mean": 0.0,
                    "mining_skipped_due_to_mixup": 1.0,
                }
            else:
                _, mined_targets, mined_stats = cost_based_pseudo_mining(
                    teacher_outputs_strong,
                    student_outputs=unsup_outputs,
                    score_threshold=self.pseudo_threshold,
                    **self.mining_cfg,
                    **mining_extra,
                )
                mined_stats.setdefault("mining_skipped_due_to_mixup", 0.0)

        mined_count = _target_count(mined_targets)
        if enable_unsupervised and self.use_mined_as_pseudo_loss and mined_count > 0:
            with torch.autocast(**autocast_kwargs):
                # REDESIGN: reuse the cached strong-view encoder features
                # (mined targets only change the decoder's DN/matching, not the
                # encoder output) -> one fewer backbone+encoder pass.
                if strong_encoded_feats is not None:
                    mined_outputs = self.student.decode_features(
                        strong_encoded_feats,
                        targets=_strip_aux_target_fields(mined_targets),
                        dn_noise_override=pseudo_dn_override,
                    )
                else:
                    mined_outputs = self.student(
                        strong_images,
                        targets=_strip_aux_target_fields(mined_targets),
                        dn_noise_override=pseudo_dn_override,
                    )
            _assert_finite_outputs(mined_outputs, "student_mined")
            loss_unsup_mined, loss_unsup_mined_dict = self._criterion_loss(mined_outputs, mined_targets)
        else:
            loss_unsup_mined = loss_unsup_high * 0.0
            loss_unsup_mined_dict = {}

        consistency_targets = _merge_targets(high_targets, mined_targets)
        # T7 (Tier 2 Phase 1): proper paired CVC that handles geometric strong
        # views. The legacy path sampled teacher (weak) features at strong-view
        # box coords -> coordinate mismatch -> CVC fell back to a noisy
        # prediction-MSE on >=75% of steps. The new path builds two query
        # dicts with the same content but each network's own view boxes, then
        # both decoders process them with consistent semantics.
        cvc_cfg = dict(self.consistency_cfg)
        cvc_pool_size = int(cvc_cfg.pop("pool_size", 7))
        cvc_min_score = float(cvc_cfg.pop("min_score", 0.0))
        cvc_warmup_epochs = int(cvc_cfg.pop("warmup_epochs", 0) or 0)
        cvc_max_pairs = int(cvc_cfg.get("max_pairs", 100))

        mixup_was_applied = mixup_stats.get("mixup_applied_count", 0.0) > 0.0
        mic_was_applied = mic_stats.get("mic_applied_count", 0.0) > 0.0
        cvc_skip_reason = 0.0
        cvc_pair_dicts: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None
        cvc_warmup_active = False
        # REDESIGN v2: skip ALL CVC work (the paired-query build + two extra
        # decoder passes per step) when the consistency loss weight is zero.
        # The paired feature-MSE CVC is a Semi-DETR transplant that measured an
        # inert ~0.005 loss here; gating it on the weight lets `consistency: 0.0`
        # truly remove the compute instead of computing a graph that is then
        # multiplied by zero. cvc_skip_reason=5 => disabled by weight.
        cvc_weight = float(self.loss_weights.get("consistency", 0.0))
        if enable_unsupervised and cvc_weight <= 0.0:
            cvc_skip_reason = 5.0
        elif enable_unsupervised:
            if self.ssod_start_epoch is not None:
                elapsed = float(self.current_epoch) - float(self.ssod_start_epoch)
                cvc_warmup_active = elapsed < float(cvc_warmup_epochs)
            if mixup_was_applied:
                # Under MixUp the student memory is a blend of two images; the
                # teacher weak features are not, so paired CVC has no clean
                # semantics. Skip CVC and let the MixUp loss carry consistency.
                cvc_skip_reason = 1.0
            elif mic_was_applied:
                # Bug #2 fix: when MIC is active the student input has 50% of
                # patches replaced by ImageNet-mean grey. CVC would then try
                # to make student's feature at masked regions match the
                # teacher's feature at the corresponding clean region -- a
                # mathematically impossible target that pushes the student
                # toward emitting zero-features for masked patches. Skip CVC
                # on MIC steps; MIC carries its own consistency signal
                # through the main detection loss on the masked input.
                cvc_skip_reason = 4.0
            elif cvc_warmup_active:
                # Literature gate: CVC on a barely-warmed teacher amplifies
                # noise. Skip for the first ``warmup_epochs`` SSOD epochs.
                cvc_skip_reason = 2.0
            elif teacher_encoded_feats is not None and hasattr(self.student, "extract_encoded_features") and hasattr(self.student, "decode_features"):
                cvc_pair_dicts = self._build_paired_consistency_queries(
                    teacher_encoded_feats,
                    weak_targets=high_targets_weak,
                    strong_targets=high_targets,
                    max_queries=cvc_max_pairs,
                    pool_size=cvc_pool_size,
                    min_score=cvc_min_score,
                )

        if enable_unsupervised and cvc_pair_dicts is not None:
            teacher_queries, student_queries = cvc_pair_dicts
            with torch.autocast(**autocast_kwargs):
                # REDESIGN: reuse cached strong-view encoder features for CVC.
                student_encoded_feats = (
                    strong_encoded_feats
                    if strong_encoded_feats is not None
                    else self.student.extract_encoded_features(strong_images)
                )
                student_consistency_outputs = self.student.decode_features(
                    student_encoded_feats,
                    targets=None,
                    consistency_queries=student_queries,
                )
            with torch.no_grad():
                with torch.autocast(**autocast_kwargs):
                    teacher_consistency_outputs = self.teacher.decode_features(
                        teacher_encoded_feats,
                        consistency_queries=teacher_queries,
                    )
            loss_consistency, consistency_stats = compute_query_consistency(
                student_consistency_outputs,
                teacher_consistency_outputs,
                **cvc_cfg,
            )
        elif enable_unsupervised:
            # No paired CVC (either skipped, or no valid pseudo-labels).
            # Emit a zero loss connected to the student graph so backward is
            # still valid even when no consistency signal exists this step.
            # Keep the zero loss connected to the graph, but accumulate in
            # fp32. Under AMP, summing all fp16 logits can overflow to inf;
            # inf * 0 then becomes NaN even though CVC was intentionally
            # skipped.
            loss_consistency = unsup_outputs["pred_logits"].float().sum() * 0.0
            consistency_stats = {"consistency_pairs": 0.0}
        else:
            loss_consistency = loss_sup * 0.0
            consistency_stats = {"consistency_pairs": 0.0}

        # REDESIGN: frozen-DINOv3 dense feature distillation on the unlabeled
        # strong view. Label-free, EMA-independent target => cannot collapse
        # onto the teacher's easy pseudo-labels (the SSOD ceiling). Skipped on
        # MixUp steps (blended image has no clean target); optionally runs on
        # MIC steps (masked student -> clean target = masked-feature distill).
        loss_distill = loss_sup * 0.0
        distill_active = (
            enable_unsupervised
            and self.distill_enabled
            and self.distill_proj is not None
            and strong_encoded_feats is not None
            and distill_input is not None
            and not mixup_was_applied
            and (self.distill_on_mic or not mic_was_applied)
        )
        if distill_active:
            with torch.no_grad():
                with torch.autocast(**autocast_kwargs):
                    t_feats = self.distill_teacher(distill_input)
                t_feat = t_feats[self.distill_level].float()
            s_feat = strong_encoded_feats[self.distill_level].float()
            s_proj = self.distill_proj(s_feat)
            if s_proj.shape[-2:] != t_feat.shape[-2:]:
                s_proj = F.interpolate(s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
            s_n = F.normalize(s_proj, dim=1)
            t_n = F.normalize(t_feat, dim=1)
            if self.distill_loss_type == "smooth_l1":
                loss_distill = F.smooth_l1_loss(s_n, t_n)
            else:  # cosine (default): 1 - per-location cosine similarity
                loss_distill = (1.0 - (s_n * t_n).sum(dim=1)).mean()

        loss_unsup = loss_unsup_high + self.mined_loss_weight * loss_unsup_mined
        total_loss = (
            self.loss_weights["supervised"] * loss_sup
            + self.loss_weights["unsupervised"] * loss_unsup
            + self.loss_weights["synthetic"] * loss_syn
            + self.loss_weights["consistency"] * loss_consistency
            + self.loss_weights["distill"] * loss_distill
        )
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm)
            optimizer.step()
        ema_delta = self.update_teacher_ema()

        loss_records.update(self._weighted_prefixed("sup", loss_sup_dict, self.loss_weights["supervised"]))
        loss_records.update(self._weighted_prefixed("syn", loss_syn_dict, self.loss_weights["synthetic"]))
        loss_records.update(self._weighted_prefixed("unsup_high", loss_unsup_high_dict, self.loss_weights["unsupervised"]))
        loss_records.update(self._weighted_prefixed("unsup_mined", loss_unsup_mined_dict, self.loss_weights["unsupervised"] * self.mined_loss_weight))
        loss_records["loss_supervised"] = loss_sup.detach()
        loss_records["loss_unsup_high"] = loss_unsup_high.detach()
        loss_records["loss_unsup_mined"] = loss_unsup_mined.detach()
        loss_records["loss_unsupervised"] = loss_unsup.detach()
        loss_records["loss_synthetic"] = loss_syn.detach()
        loss_records["loss_synthetic_weighted"] = (self.loss_weights["synthetic"] * loss_syn).detach()
        loss_records["loss_consistency"] = loss_consistency.detach()
        loss_records["loss_distill"] = loss_distill.detach()
        loss_records["loss_distill_weighted"] = (self.loss_weights["distill"] * loss_distill).detach()
        loss_records["distill_active"] = torch.as_tensor(float(distill_active), device=device)
        loss_records["verifier_added"] = torch.as_tensor(float(self._verif_added_last), device=device)
        loss_records["verifier_kept"] = torch.as_tensor(float(self._verif_kept_last), device=device)
        loss_records["verifier_dropped"] = torch.as_tensor(float(self._verif_dropped_last), device=device)
        loss_records["soft_bg_regions"] = torch.as_tensor(float(self._soft_bg_count_last), device=device)
        loss_records["loss_total"] = total_loss.detach()
        loss_records["high_conf_labels"] = torch.as_tensor(high_label_count, device=device, dtype=torch.float32)
        loss_records["mined_labels"] = torch.as_tensor(mined_stats["mined_labels"], device=device)
        loss_records["pseudo_label_count"] = torch.as_tensor(
            high_label_count + mined_stats["mined_labels"],
            device=device,
        )
        loss_records["pseudo_label_quality"] = _target_score_mean(consistency_targets, device)
        loss_records["consistency_pairs"] = torch.as_tensor(consistency_stats["consistency_pairs"], device=device)
        # T7 diagnostics: which CVC path actually ran this step.
        # 0 = paired CVC ran, 1 = skipped (MixUp), 2 = skipped (SSOD warmup),
        # 3 = skipped (no paired queries / pseudo_labels were empty).
        cvc_path_code = float(cvc_skip_reason)
        if enable_unsupervised and cvc_pair_dicts is None and cvc_skip_reason == 0.0:
            cvc_path_code = 3.0
        loss_records["cvc_skip_reason"] = torch.as_tensor(cvc_path_code, device=device)
        loss_records["cvc_active"] = torch.as_tensor(
            float(enable_unsupervised and cvc_pair_dicts is not None), device=device
        )
        # T9 diagnostics.
        loss_records["pseudo_dn_enabled"] = torch.as_tensor(float(self.pseudo_dn_enabled), device=device)
        loss_records["pseudo_dn_label_noise"] = torch.as_tensor(float(self.pseudo_dn_label_noise), device=device)
        loss_records["pseudo_dn_box_noise"] = torch.as_tensor(float(self.pseudo_dn_box_noise), device=device)
        loss_records["pseudo_dn_min_score"] = torch.as_tensor(float(self.pseudo_dn_min_score), device=device)
        loss_records["pseudo_dn_kept_count"] = torch.as_tensor(float(pseudo_dn_kept), device=device)
        loss_records["pseudo_dn_total_count"] = torch.as_tensor(float(pseudo_dn_total), device=device)
        loss_records["unsupervised_enabled"] = torch.as_tensor(float(enable_unsupervised), device=device)
        loss_records["mined_loss_enabled"] = torch.as_tensor(float(enable_unsupervised and self.use_mined_as_pseudo_loss), device=device)
        loss_records["teacher_ema_delta"] = ema_delta.to(device)
        loss_records["amp_enabled"] = torch.as_tensor(float(amp_enabled), device=device)
        loss_records["labeled_mixup_target_count"] = _target_flag_sum(labeled_targets, "_mixup_applied", device)
        loss_records["labeled_copyblend_target_count"] = _target_flag_sum(labeled_targets, "_copyblend_applied", device)
        loss_records["labeled_mosaic_target_count"] = _target_flag_sum(labeled_targets, "_mosaic_applied", device)
        for key in (
            "teacher_score_mean",
            "teacher_score_p50",
            "teacher_score_p75",
            "teacher_score_p90",
            "teacher_score_p95",
            "teacher_score_max",
            "high_conf_effective_threshold",
            "high_conf_target_weight_mean",
        ):
            if key in high_stats:
                loss_records[key] = torch.as_tensor(float(high_stats[key]), device=device)
        for key in (
            "mined_target_weight_mean",
            "candidate_initial_labels",
            "candidate_query_labels",
            "matched_pairs",
            "accepted_mined_labels",
            "mined_acceptance_rate",
            "mining_skipped_due_to_mixup",
        ):
            if key in mined_stats:
                loss_records[f"mining_{key}"] = torch.as_tensor(float(mined_stats[key]), device=device)
        # Per-class pseudo-label diagnostics -- essential for spotting the
        # tail-class starvation symptom that motivated T1.
        if self.num_classes is not None:
            for cls_id in range(int(self.num_classes)):
                hi_key = f"high_conf_labels_class_{cls_id}"
                loss_records[hi_key] = torch.as_tensor(
                    float(_target_class_count(high_targets, cls_id)),
                    device=device,
                )
                if hi_key in high_stats:
                    loss_records[f"raw_{hi_key}"] = torch.as_tensor(float(high_stats[hi_key]), device=device)
                mi_key = f"mined_labels_class_{cls_id}"
                loss_records[mi_key] = torch.as_tensor(
                    float(_target_class_count(mined_targets, cls_id)),
                    device=device,
                )
                if mi_key in mined_stats:
                    loss_records[f"raw_{mi_key}"] = torch.as_tensor(float(mined_stats[mi_key]), device=device)
        # T2 MixUp + T4 EMA decay + T1 adaptive threshold + T8 MIC diagnostics.
        for key, value in mixup_stats.items():
            loss_records[key] = torch.as_tensor(float(value), device=device)
        for key, value in mic_stats.items():
            loss_records[key] = torch.as_tensor(float(value), device=device)
        loss_records["ema_decay_current"] = torch.as_tensor(float(self.ema_decay), device=device)
        if self.adaptive_threshold is not None:
            for key, value in self.adaptive_threshold.diagnostics().items():
                loss_records[key] = torch.as_tensor(float(value), device=device)
        for key, value in pair_aug.aug_stats.items():
            loss_records[key] = torch.as_tensor(float(value), device=device)
        for key, value in mapping_stats.items():
            loss_records[key] = torch.as_tensor(float(value), device=device)
        return loss_records
