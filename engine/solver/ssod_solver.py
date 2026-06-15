"""Solver for DEIMv2 semi-supervised object detection."""

import csv
import json
import math
import time
from itertools import cycle

import torch
import torchvision

from ..deim import SSODTrainer
from ..deim.adaptive_threshold import ClassAdaptivePseudoThreshold
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ._solver import BaseSolver
from .det_engine import evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _linear_value(start: float, end: float, index: int, length: int) -> float:
    if length <= 0:
        return end
    factor = min(max((index + 1) / float(length), 0.0), 1.0)
    return start + (end - start) * factor


def resolve_synthetic_schedule(
    ssod_cfg: dict,
    epoch: int,
    actual_unsup_start_epoch=None,
    default_weight: float = 0.0,
):
    schedule = dict(ssod_cfg.get("synthetic_schedule") or {})
    if not schedule.get("enabled", False):
        weight = float(default_weight)
        return {
            "synthetic_loss_weight": weight,
            "target_synthetic_loss_weight": weight,
            "synthetic_enabled": weight > 0.0,
            "synthetic_stop_reason": 0.0,
            "synthetic_close_loader_when_zero": 0.0,
        }

    mode = str(schedule.get("mode", "burnin_until_unsup"))
    active_weight = float(schedule.get("active_weight", default_weight))
    after_stop_weight = float(schedule.get("after_stop_weight", 0.0))
    max_active_epoch = schedule.get("max_active_epoch", None)
    close_loader = bool(schedule.get("close_loader_when_zero", True))

    stop_reason = 0.0
    should_stop = False
    if mode == "burnin_until_unsup":
        if actual_unsup_start_epoch is not None and epoch >= int(actual_unsup_start_epoch):
            should_stop = True
            stop_reason = 1.0
        if max_active_epoch is not None and epoch >= int(max_active_epoch):
            should_stop = True
            if stop_reason == 0.0:
                stop_reason = 2.0
    elif mode == "fixed_until_epoch":
        if max_active_epoch is not None and epoch >= int(max_active_epoch):
            should_stop = True
            stop_reason = 2.0
    else:
        raise ValueError(f"Unsupported ssod.synthetic_schedule.mode={mode!r}")

    weight = after_stop_weight if should_stop else active_weight
    return {
        "synthetic_loss_weight": weight,
        "target_synthetic_loss_weight": active_weight,
        "synthetic_enabled": weight > 0.0,
        "synthetic_stop_reason": stop_reason,
        "synthetic_close_loader_when_zero": float(close_loader),
    }


def _default_threshold_sweep_values():
    # DEIM/DETR focal scores are often useful at much lower absolute values than
    # the 0.05..0.95 grid used by two-stage detectors. Keep the old grid, but add
    # a dense low-score region so F1 diagnostics do not collapse to all-zero rows.
    low = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075]
    mid = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    high_mid = [round(0.06 + i * 0.02, 2) for i in range(8)]
    high = [round(0.25 + i * 0.05, 2) for i in range(15)]
    return sorted(set(low + mid + high_mid + high))


class SSODDetSolver(BaseSolver):
    def train(self):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.labeled_dataloader = self._build_loader("labeled_dataloader")
        self.unlabeled_dataloader = self._build_loader("unlabeled_dataloader")
        self.synthetic_dataloader = None
        ssod_cfg = self.cfg.yaml_cfg.get("ssod", {})
        syn_weight = float(ssod_cfg.get("losses", {}).get("synthetic", 0.0))
        if "synthetic_dataloader" in self.cfg.yaml_cfg and syn_weight > 0:
            self.synthetic_dataloader = self._build_loader("synthetic_dataloader")

        self.val_dataloader = None
        if "val_dataloader" in self.cfg.yaml_cfg:
            self.val_dataloader = dist_utils.warp_loader(
                self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
            )
            self.evaluator = self.cfg.evaluator

        self.self_lr_scheduler = False
        if self.cfg.lrsheduler is not None:
            iter_per_epoch = min(len(self.labeled_dataloader), len(self.unlabeled_dataloader))
            print("     ## Using Self-defined Scheduler-{} ## ".format(self.cfg.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                self.cfg.lr_gamma,
                iter_per_epoch,
                total_epochs=self.cfg.epoches,
                warmup_iter=self.cfg.warmup_iter,
                flat_epochs=self.cfg.flat_epoch,
                no_aug_epochs=self.cfg.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.load_resume_state(self.cfg.resume)

    def _build_loader(self, name: str):
        loader = self.cfg.build_dataloader(name)
        return dist_utils.warp_loader(loader, shuffle=loader.shuffle)

    def _make_trainer(self):
        ssod_cfg = self.cfg.yaml_cfg.get("ssod", {})
        num_classes = int(self.cfg.yaml_cfg.get("num_classes", 0) or 0) or None
        augmentation_cfg = dict(ssod_cfg.get("augmentation") or {})
        # T2 MixPL knobs live alongside augmentation in the config but they are
        # not parameters of SSODPairAugmentor -- split them out before forwarding.
        mixup_prob = float(augmentation_cfg.pop("mixup_prob", 0.0))
        mixup_alpha = float(augmentation_cfg.pop("mixup_alpha", 0.5))
        # T8 MIC knobs also live under augmentation; split them out the same way.
        mic_prob = float(augmentation_cfg.pop("mic_prob", 0.0))
        mic_patch_size = int(augmentation_cfg.pop("mic_patch_size", 32))
        mic_mask_ratio = float(augmentation_cfg.pop("mic_mask_ratio", 0.5))
        mic_fill_value = augmentation_cfg.pop("mic_fill_value", None)

        adaptive_cfg = dict(ssod_cfg.get("adaptive_threshold") or {})
        adaptive_threshold = None
        if bool(adaptive_cfg.get("enabled", False)) and num_classes:
            init_thr_default = float(ssod_cfg.get("pseudo_threshold", 0.1) or 0.1)
            adaptive_threshold = ClassAdaptivePseudoThreshold(
                num_classes=num_classes,
                target_quantile_per_class=adaptive_cfg.get("target_quantile_per_class"),
                init_thresholds=adaptive_cfg.get("init_thresholds"),
                default_threshold=float(adaptive_cfg.get("default_threshold", init_thr_default)),
                default_quantile=float(adaptive_cfg.get("default_quantile", 0.95)),
                min_thresh=float(adaptive_cfg.get("min_thresh", 0.01)),
                max_thresh=float(adaptive_cfg.get("max_thresh", 0.5)),
                history_size=int(adaptive_cfg.get("history_size", 2000)),
                min_history_for_quantile=int(adaptive_cfg.get("min_history_for_quantile", 100)),
                update_interval=int(adaptive_cfg.get("update_interval", 50)),
                enabled=True,
                input_score_floor=float(adaptive_cfg.get("input_score_floor", 0.0)),
            )
            print(
                f"SSOD adaptive threshold enabled: num_classes={num_classes} "
                f"quantiles={adaptive_threshold.target_quantile_per_class} "
                f"init={adaptive_threshold.init_thresholds}"
            )

        trainer = SSODTrainer(
            student_model=self.model,
            criterion=self.criterion,
            ema_decay=ssod_cfg.get("teacher_ema_decay", 0.9996),
            pseudo_threshold=ssod_cfg.get("pseudo_threshold", 0.7),
            loss_weights=ssod_cfg.get("losses", None),
            mining_cfg=ssod_cfg.get("mining", None),
            consistency_cfg=ssod_cfg.get("consistency", None),
            augmentation_cfg=augmentation_cfg,
            adaptive_threshold=adaptive_threshold,
            num_classes=num_classes,
            ema_decay_warmup_start=float(ssod_cfg.get("teacher_ema_warmup_decay", 0.99)),
            ema_decay_warmup_epochs=int(ssod_cfg.get("teacher_ema_warmup_epochs", 0) or 0),
            ema_decay_burnin=(
                float(ssod_cfg["teacher_ema_burnin_decay"])
                if ssod_cfg.get("teacher_ema_burnin_decay") is not None
                else None
            ),
            ema_decay_warmup_curve=str(ssod_cfg.get("teacher_ema_warmup_curve", "cosine")),
            mixup_prob=mixup_prob,
            mixup_alpha=mixup_alpha,
            mic_prob=mic_prob,
            mic_patch_size=mic_patch_size,
            mic_mask_ratio=mic_mask_ratio,
            mic_fill_value=mic_fill_value,
            pseudo_dn_cfg=ssod_cfg.get("pseudo_dn"),
            distill_cfg=ssod_cfg.get("distill"),
            verifier_cfg=ssod_cfg.get("verifier"),
            soft_bg_cfg=ssod_cfg.get("soft_background"),
        ).to(self.device)

        # REDESIGN: register the distillation projection conv with the optimizer.
        # The base optimizer + custom LR scheduler were built from the student's
        # param groups before the trainer existed; the scheduler caches one
        # base_lr per group, so ADDING a new group breaks its indexing. Instead
        # we append the proj params to the existing default (head) param group
        # -- it trains at the head lr/wd and the group count stays unchanged.
        if getattr(trainer, "distill_proj", None) is not None:
            proj_params = [p for p in trainer.distill_proj.parameters() if p.requires_grad]
            self.optimizer.param_groups[-1]["params"].extend(proj_params)
            print(
                f"[REDESIGN] appended {len(proj_params)} distill_proj tensors to "
                f"optimizer group[-1] (lr={self.optimizer.param_groups[-1].get('lr')})"
            )

        # Resume fix: the base solver only restores attributes it can find on
        # ``self`` at load time, but the trainer (and thus the teacher / the
        # adaptive-threshold tracker) is only constructed after the resume
        # call has already finished. Without the lines below, a resumed SSOD
        # run silently re-initialises ``trainer.teacher`` as a deepcopy of the
        # student, throwing away every EMA step performed before the resume.
        if self.cfg.resume:
            try:
                resume_state = torch.load(self.cfg.resume, map_location="cpu")
            except Exception as exc:
                print(f"[SSOD] Could not re-open resume checkpoint to recover teacher state: {exc}")
                resume_state = None
            if isinstance(resume_state, dict):
                teacher_state = resume_state.get("teacher")
                if teacher_state:
                    try:
                        dist_utils.de_parallel(trainer.teacher).load_state_dict(teacher_state)
                        print("[SSOD] Restored teacher EMA weights from resume checkpoint.")
                    except Exception as exc:
                        print(f"[SSOD] Failed to restore teacher EMA: {exc}")
                if adaptive_threshold is not None:
                    adaptive_state = resume_state.get("adaptive_threshold")
                    if adaptive_state:
                        try:
                            adaptive_threshold.load_state_dict(adaptive_state)
                            print(
                                "[SSOD] Restored adaptive threshold history "
                                f"(total_updates={adaptive_state.get('total_updates', 0)})"
                            )
                        except Exception as exc:
                            print(f"[SSOD] Failed to restore adaptive threshold state: {exc}")
        return trainer

    def fit(self):
        self.train()
        args = self.cfg
        ssod_cfg = args.yaml_cfg.get("ssod", {})
        max_steps = int(ssod_cfg.get("max_steps", args.yaml_cfg.get("max_steps", 0)) or 0)
        trainer = self._make_trainer()
        if max_steps > 0:
            return self._fit_steps(trainer, max_steps)

        start_time = time.time()
        best_metric = -1.0
        best_metric_name = str(ssod_cfg.get("best_metric", "coco_bbox_mAP50"))
        best_epoch = -1
        patience = int(ssod_cfg.get("early_stop_patience", 0) or 0)
        min_epochs = int(ssod_cfg.get("min_epochs", 0) or 0)
        checkpoint_period = int(ssod_cfg.get("checkpoint_period", 0) or 0)
        bad_epochs = 0
        start_epoch = self.last_epoch + 1
        self._ssod_last_val_stats = {}
        self._ssod_unsup_start_epoch = None

        # Preserve SSOD state on resume. Without this, metric-gated unsupervised
        # training can be disabled after interruption and the next validation can
        # overwrite a stronger best_teacher_ema checkpoint with a weaker one.
        history_rows = []
        metrics_path = self.output_dir / "epoch_metrics.jsonl" if self.output_dir else None
        if metrics_path is not None and metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if int(row.get("epoch", -1)) <= self.last_epoch:
                        history_rows.append(row)
        if history_rows:
            latest_row = max(history_rows, key=lambda row: int(row.get("epoch", -1)))
            self._ssod_last_val_stats = dict(latest_row.get("val") or {})
            for hist_row in sorted(history_rows, key=lambda row: int(row.get("epoch", -1)), reverse=True):
                hist_train = hist_row.get("train") or {}
                schedule_start = hist_train.get("unsupervised_schedule_start_epoch")
                was_unsupervised = (
                    float(hist_train.get("unsupervised_loss_weight", 0.0) or 0.0) > 0.0
                    or bool(hist_train.get("enable_unsupervised", False))
                )
                if schedule_start is not None and was_unsupervised:
                    self._ssod_unsup_start_epoch = int(float(schedule_start))
                    break

            for row in history_rows:
                val = row.get("val") or {}
                metric = float(val.get(best_metric_name, val.get("macro_f1_iou50", 0.0)) or 0.0)
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = int(row.get("epoch", -1))
            if best_epoch >= 0:
                bad_epochs = sum(1 for row in history_rows if int(row.get("epoch", -1)) > best_epoch)
                print(
                    f"Restored SSOD history: best_epoch={best_epoch}, "
                    f"best_{best_metric_name}={best_metric:.6f}, bad_epochs={bad_epochs}, "
                    f"last_val_epoch={latest_row.get('epoch')}, "
                    f"unsup_start_epoch={self._ssod_unsup_start_epoch}"
                )

        if self.cfg.resume and (not self._ssod_last_val_stats or self._ssod_unsup_start_epoch is None):
            try:
                resume_state = torch.load(self.cfg.resume, map_location="cpu")
            except Exception:
                resume_state = {}
            if isinstance(resume_state, dict):
                if not self._ssod_last_val_stats:
                    self._ssod_last_val_stats = dict(resume_state.get("ssod_val_stats") or {})
                train_state = resume_state.get("ssod_train_stats") or {}
                schedule_start = train_state.get("unsupervised_schedule_start_epoch")
                was_unsupervised = (
                    float(train_state.get("unsupervised_loss_weight", 0.0) or 0.0) > 0.0
                    or bool(train_state.get("enable_unsupervised", False))
                )
                if self._ssod_unsup_start_epoch is None and schedule_start is not None and was_unsupervised:
                    self._ssod_unsup_start_epoch = int(float(schedule_start))

        for epoch in range(start_epoch, int(args.epoches)):
            train_stats = self._train_one_epoch(trainer, epoch)
            self.last_epoch = epoch

            val_stats = {}
            if self.val_dataloader is not None:
                val_stats = self._evaluate_teacher(trainer, epoch)
                metric = float(val_stats.get(best_metric_name, val_stats.get("macro_f1_iou50", 0.0)))
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    bad_epochs = 0
                    if self.output_dir and dist_utils.is_main_process():
                        dist_utils.save_on_master(
                            self._ssod_state_dict(trainer, epoch, train_stats, val_stats),
                            self.output_dir / "best_teacher_ema.pth",
                        )
                else:
                    bad_epochs += 1
                self._ssod_last_val_stats = dict(val_stats)

            if self.output_dir and dist_utils.is_main_process():
                dist_utils.save_on_master(
                    self._ssod_state_dict(trainer, epoch, train_stats, val_stats),
                    self.output_dir / "last_ssod.pth",
                )
                if checkpoint_period > 0 and (epoch + 1) % checkpoint_period == 0:
                    dist_utils.save_on_master(
                        self._ssod_state_dict(trainer, epoch, train_stats, val_stats),
                        self.output_dir / f"checkpoint_ssod_{epoch + 1:04d}.pth",
                    )
                with (self.output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric_name,
                        "best_metric_value": best_metric,
                        "train": train_stats,
                        "val": val_stats,
                    }) + "\n")

            if patience > 0 and (epoch + 1) >= min_epochs and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_{best_metric_name}={best_metric:.6f}")
                break

        total_time = time.time() - start_time
        print(f"SSOD epoch training time {total_time:.1f}s")
        return {"best_epoch": best_epoch, "best_metric": best_metric_name, "best_metric_value": best_metric}

    def _step_lr_schedulers(self, current_iter: int = 0):
        if getattr(self, "self_lr_scheduler", False):
            self.optimizer = self.lr_scheduler.step(current_iter, self.optimizer)
        elif self.lr_warmup_scheduler is not None and not self.lr_warmup_scheduler.finished():
            self.lr_warmup_scheduler.step()
        elif self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _collate_aug_stats(self):
        stats = {}
        for prefix, loader in (
            ("labeled", self.labeled_dataloader),
            ("unlabeled", self.unlabeled_dataloader),
            ("synthetic", self.synthetic_dataloader),
        ):
            if loader is None:
                continue
            collate = getattr(loader, "collate_fn", None)
            if collate is None:
                continue
            for attr in ("last_mixup_applied", "last_copyblend_applied", "mixup_applied_count", "copyblend_applied_count"):
                if hasattr(collate, attr):
                    stats[f"{prefix}_{attr}"] = float(getattr(collate, attr))
            transforms = getattr(getattr(loader, "dataset", None), "_transforms", None)
            for attr in ("last_mosaic_applied", "mosaic_applied_count"):
                if hasattr(transforms, attr):
                    stats[f"{prefix}_{attr}"] = float(getattr(transforms, attr))
        return stats

    def _apply_epoch_schedule(self, trainer: SSODTrainer, epoch: int):
        ssod_cfg = self.cfg.yaml_cfg.get("ssod", {})
        losses = ssod_cfg.get("losses", {})
        burn_in_epochs = int(ssod_cfg.get("burn_in_epochs", 0) or 0)
        time_ready = epoch >= burn_in_epochs

        gate_name = str(ssod_cfg.get("unsupervised_start_metric", "") or "")
        gate_min_raw = ssod_cfg.get("unsupervised_start_min_metric", None)
        gate_ready = True
        gate_value = -1.0
        if time_ready and gate_name and gate_min_raw is not None:
            last_val = getattr(self, "_ssod_last_val_stats", {}) or {}
            gate_value = float(last_val.get(gate_name, -1.0))
            gate_ready = gate_value >= float(gate_min_raw)

        actual_start = getattr(self, "_ssod_unsup_start_epoch", None)
        # The metric gate only decides the first SSOD start. Once unsupervised
        # training has begun, do not turn it off because of single-val noise.
        enable_unsupervised = time_ready and (gate_ready or actual_start is not None)
        # Late-stage shutdown: at the aug-off boundary (no_aug / collate stop_epoch)
        # the strong augmentation disappears, the student becomes overconfident on
        # clean images, the EMA teacher's score distribution spikes, and a FIXED
        # pseudo_threshold lets a flood of pseudo-labels (and soft-bg ignore regions)
        # through -> loss explosion and a val-mAP collapse (observed in v6 at the
        # exact stop_epoch). Turning the unsupervised branch off at that boundary
        # makes the no-aug tail a clean supervised polish (standard DEIM behaviour)
        # and removes the collapse entirely. Defaults to never (None) when unset.
        unsup_stop_epoch = ssod_cfg.get("unsupervised_stop_epoch", None)
        if unsup_stop_epoch is not None and epoch >= int(unsup_stop_epoch):
            if enable_unsupervised:
                print(
                    f"Disable SSOD unsupervised branch at epoch {epoch} "
                    f"(unsupervised_stop_epoch={int(unsup_stop_epoch)}; no-aug supervised tail)"
                )
            enable_unsupervised = False
        if enable_unsupervised and actual_start is None:
            self._ssod_unsup_start_epoch = epoch
            actual_start = epoch
            print(f"Enable SSOD unsupervised branch at epoch {epoch}")
        # T4: surface the unsupervised start epoch to the trainer so its EMA
        # decay warmup can begin counting from the right reference point even
        # when training is resumed mid-run.
        if actual_start is not None and getattr(trainer, "ssod_start_epoch", None) != int(actual_start):
            trainer.ssod_start_epoch = int(actual_start)

        schedule_from_actual = bool(ssod_cfg.get("schedule_from_actual_unsup_start", False))
        schedule_start_epoch = actual_start if schedule_from_actual and actual_start is not None else burn_in_epochs

        target_unsup = float(losses.get("unsupervised", trainer.loss_weights.get("unsupervised", 1.0)))
        unsup_ramp_epochs = int(ssod_cfg.get("unsupervised_ramp_epochs", 0) or 0)
        if not enable_unsupervised:
            unsup_weight = 0.0
        elif unsup_ramp_epochs > 0:
            unsup_weight = _linear_value(0.0, target_unsup, epoch - schedule_start_epoch, unsup_ramp_epochs)
        else:
            unsup_weight = target_unsup

        pseudo_start = float(ssod_cfg.get("pseudo_threshold", trainer.pseudo_threshold))
        pseudo_final = float(ssod_cfg.get("pseudo_threshold_final", pseudo_start))
        pseudo_ramp_epochs = int(ssod_cfg.get("pseudo_threshold_warmup_epochs", 0) or 0)
        pseudo_ramp_start = int(ssod_cfg.get("pseudo_threshold_warmup_start_epoch", burn_in_epochs) or 0)
        if schedule_from_actual and actual_start is not None:
            pseudo_ramp_start = schedule_start_epoch
        # T1: when adaptive per-class thresholds are active they own the
        # threshold knob; the legacy linear warmup would otherwise re-raise the
        # fallback ``pseudo_threshold`` (used only for unseen classes) and
        # confuse logs.
        adaptive_active = getattr(trainer, "adaptive_threshold", None) is not None
        if adaptive_active:
            pseudo_threshold = pseudo_start
        elif epoch < pseudo_ramp_start:
            pseudo_threshold = pseudo_start
        else:
            pseudo_threshold = _linear_value(
                pseudo_start,
                pseudo_final,
                epoch - pseudo_ramp_start,
                pseudo_ramp_epochs,
            )

        trainer.pseudo_threshold = pseudo_threshold
        trainer.loss_weights["unsupervised"] = unsup_weight
        synthetic_schedule = resolve_synthetic_schedule(
            ssod_cfg,
            epoch,
            actual_unsup_start_epoch=actual_start,
            default_weight=float(losses.get("synthetic", trainer.loss_weights.get("synthetic", 0.0))),
        )
        trainer.loss_weights["synthetic"] = float(synthetic_schedule["synthetic_loss_weight"])
        trainer.current_epoch = epoch

        restart_epoch = getattr(self.labeled_dataloader.collate_fn, "stop_epoch", None)
        restart_decay = ssod_cfg.get(
            "teacher_ema_restart_decay",
            getattr(self.labeled_dataloader.collate_fn, "ema_restart_decay", None),
        )
        if restart_epoch is not None and epoch == int(restart_epoch) and restart_decay is not None:
            # The EMA warmup added in T4 owns ``ema_decay`` on every step. Bump
            # the target (durable) value so the late-stage restart still takes
            # effect, and disable any remaining warmup so we do not ramp down
            # again accidentally.
            trainer.ema_decay_target = float(restart_decay)
            trainer.ema_decay = float(restart_decay)
            trainer.ema_decay_warmup_epochs = 0
            print(f"Refresh SSOD teacher EMA at epoch {epoch} with decay {trainer.ema_decay_target}")

        return {
            "enable_unsupervised": enable_unsupervised,
            "pseudo_threshold": pseudo_threshold,
            "unsupervised_loss_weight": unsup_weight,
            "target_unsupervised_loss_weight": target_unsup,
            "unsupervised_metric_gate_ready": float(gate_ready),
            "unsupervised_metric_gate_value": gate_value,
            "unsupervised_schedule_start_epoch": float(schedule_start_epoch),
            **synthetic_schedule,
        }

    def _fit_steps(self, trainer: SSODTrainer, max_steps: int):
        args = self.cfg
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        print_freq = args.print_freq or 10
        schedule = self._apply_epoch_schedule(trainer, 0)
        enable_unsupervised = schedule["enable_unsupervised"]
        synthetic_active = (
            self.synthetic_dataloader is not None
            and bool(schedule.get("synthetic_enabled", False))
            and float(schedule.get("synthetic_loss_weight", 0.0)) > 0.0
        )
        labeled_iter = _cycle(self.labeled_dataloader)
        unlabeled_iter = _cycle(self.unlabeled_dataloader)
        synthetic_iter = _cycle(self.synthetic_dataloader) if synthetic_active else None

        start_time = time.time()
        for step in range(max_steps):
            trainer.current_epoch = 0
            loss_dict = trainer.train_step(
                next(labeled_iter),
                next(unlabeled_iter),
                next(synthetic_iter) if synthetic_iter is not None else None,
                optimizer=self.optimizer,
                device=self.device,
                scaler=self.scaler,
                max_norm=args.clip_max_norm,
                enable_unsupervised=enable_unsupervised,
            )
            for name, value in self._collate_aug_stats().items():
                loss_dict[name] = torch.as_tensor(value, device=self.device)
            for name, value in schedule.items():
                if isinstance(value, (bool, int, float)):
                    loss_dict[name] = torch.as_tensor(float(value), device=self.device)

            self._step_lr_schedulers(step)

            reduced = dist_utils.reduce_dict(loss_dict)
            total = reduced["loss_total"]
            if not math.isfinite(float(total)):
                raise RuntimeError(f"Non-finite SSOD loss at step {step}: {reduced}")

            metric_logger.update(**reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            if self.writer and dist_utils.is_main_process():
                global_step = step + 1
                self.writer.add_scalar("SSOD/Loss_total", total.item(), global_step)
                for name, value in reduced.items():
                    self.writer.add_scalar(f"SSOD/{name}", value.item(), global_step)

            if step == 0 or (step + 1) % print_freq == 0 or step + 1 == max_steps:
                print(f"SSOD step [{step + 1}/{max_steps}]  {metric_logger}")

        metric_logger.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if torch.cuda.is_available():
            stats["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "ssod_log.txt").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"steps": max_steps, **stats}) + "\n")
            dist_utils.save_on_master(self.state_dict(), self.output_dir / "last_ssod.pth")
        print(f"SSOD training time {time.time() - start_time:.1f}s")
        print("SSOD averaged stats:", stats)
        return stats

    def _train_one_epoch(self, trainer: SSODTrainer, epoch: int):
        if hasattr(self.labeled_dataloader, "set_epoch"):
            self.labeled_dataloader.set_epoch(epoch)
        if hasattr(self.unlabeled_dataloader, "set_epoch"):
            self.unlabeled_dataloader.set_epoch(epoch)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        print_freq = self.cfg.print_freq or 20
        schedule = self._apply_epoch_schedule(trainer, epoch)
        enable_unsupervised = schedule["enable_unsupervised"]
        synthetic_active = (
            self.synthetic_dataloader is not None
            and bool(schedule.get("synthetic_enabled", False))
            and float(schedule.get("synthetic_loss_weight", 0.0)) > 0.0
        )
        if synthetic_active and hasattr(self.synthetic_dataloader, "set_epoch"):
            self.synthetic_dataloader.set_epoch(epoch)

        steps = min(len(self.labeled_dataloader), len(self.unlabeled_dataloader))
        if synthetic_active:
            steps = min(steps, len(self.synthetic_dataloader))

        labeled_iter = _cycle(self.labeled_dataloader)
        unlabeled_iter = _cycle(self.unlabeled_dataloader)
        synthetic_iter = _cycle(self.synthetic_dataloader) if synthetic_active else None

        for step in range(steps):
            loss_dict = trainer.train_step(
                next(labeled_iter),
                next(unlabeled_iter),
                next(synthetic_iter) if synthetic_iter is not None else None,
                optimizer=self.optimizer,
                device=self.device,
                scaler=self.scaler,
                max_norm=self.cfg.clip_max_norm,
                enable_unsupervised=enable_unsupervised,
            )
            for name, value in self._collate_aug_stats().items():
                loss_dict[name] = torch.as_tensor(value, device=self.device)
            self._step_lr_schedulers(epoch * steps + step)

            for name, value in schedule.items():
                if isinstance(value, (bool, int, float)):
                    loss_dict[name] = torch.as_tensor(float(value), device=self.device)
            reduced = dist_utils.reduce_dict(loss_dict)
            total = reduced["loss_total"]
            if not math.isfinite(float(total)):
                raise RuntimeError(f"Non-finite SSOD loss at epoch {epoch}, step {step}: {reduced}")
            metric_logger.update(**reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            if step == 0 or (step + 1) % print_freq == 0 or step + 1 == steps:
                print(f"SSOD epoch [{epoch + 1}/{self.cfg.epoches}] step [{step + 1}/{steps}]  {metric_logger}")

        metric_logger.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if torch.cuda.is_available():
            stats["cuda_max_mem_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        return stats

    @torch.no_grad()
    def _class_names(self):
        dataset = getattr(self.val_dataloader, "dataset", None)
        category2name = getattr(dataset, "category2name", None)
        if isinstance(category2name, dict):
            return {int(k): v for k, v in category2name.items()}
        return {idx: f"class_{idx}" for idx in range(int(getattr(self.postprocessor, "num_classes", 0)))}

    def _score_from_counts(self, tp: int, fp: int, fn: int):
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1

    def _match_counts(self, pred_boxes, pred_scores, gt_boxes, threshold: float, iou_threshold: float):
        keep = pred_scores >= threshold
        boxes = pred_boxes[keep]
        matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=self.device)
        tp = 0
        fp = 0
        if boxes.numel() > 0 and gt_boxes.numel() > 0:
            ious = torchvision.ops.box_iou(boxes, gt_boxes)
        else:
            ious = None
        for pred_idx in range(boxes.shape[0]):
            if ious is None or gt_boxes.shape[0] == 0:
                fp += 1
                continue
            best_iou, best_gt = ious[pred_idx].max(dim=0)
            if best_iou >= iou_threshold and not matched[best_gt]:
                matched[best_gt] = True
                tp += 1
            else:
                fp += 1
        fn = int(gt_boxes.shape[0] - matched.sum().item())
        return tp, fp, fn

    def _target_boxes_to_original_xyxy(self, target, class_id: int):
        gt_mask = target["labels"] == class_id
        gt_boxes = target["boxes"][gt_mask]
        if gt_boxes.numel() == 0:
            return gt_boxes.reshape(0, 4).to(dtype=torch.float32)

        box_format = getattr(gt_boxes, "format", None)
        box_format = getattr(box_format, "value", box_format)
        box_format = str(box_format).lower() if box_format is not None else None
        canvas_size = getattr(gt_boxes, "canvas_size", None)
        gt_boxes = gt_boxes.to(dtype=torch.float32)

        if box_format and box_format != "xyxy":
            gt_boxes = torchvision.ops.box_convert(gt_boxes, in_fmt=box_format, out_fmt="xyxy")
        elif box_format is None and float(gt_boxes.detach().max().item()) <= 1.5:
            gt_boxes = torchvision.ops.box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        orig_size = target["orig_size"].to(gt_boxes.device, dtype=gt_boxes.dtype)
        orig_scale = orig_size.repeat(2)
        if float(gt_boxes.detach().max().item()) <= 1.5:
            return gt_boxes * orig_scale

        if canvas_size is not None:
            canvas_h, canvas_w = [float(x) for x in canvas_size]
            canvas_scale = torch.as_tensor(
                [canvas_w, canvas_h, canvas_w, canvas_h],
                device=gt_boxes.device,
                dtype=gt_boxes.dtype,
            ).clamp_min(1.0)
            return gt_boxes * (orig_scale / canvas_scale)

        return gt_boxes

    def _threshold_sweep(self, model, thresholds=None, iou_threshold: float = 0.5):
        if thresholds is None:
            thresholds = _default_threshold_sweep_values()
        model.eval()
        self.postprocessor.eval()
        num_classes = int(getattr(self.postprocessor, "num_classes", 0))
        class_names = self._class_names()
        rows = {
            (class_id, thr): {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
                "threshold": thr,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pred": 0,
                "gt": 0,
            }
            for class_id in range(num_classes)
            for thr in thresholds
        }

        for samples, targets in self.val_dataloader:
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
            outputs = model(samples)
            orig_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
            results = self.postprocessor(outputs, orig_sizes)

            for result, target in zip(results, targets):
                for class_id in range(num_classes):
                    pred_mask = result["labels"] == class_id
                    pred_boxes = result["boxes"][pred_mask]
                    pred_scores = result["scores"][pred_mask]

                    gt_boxes = self._target_boxes_to_original_xyxy(target, class_id)

                    order = torch.argsort(pred_scores, descending=True)
                    pred_boxes = pred_boxes[order]
                    pred_scores = pred_scores[order]

                    for thr in thresholds:
                        tp, fp, fn = self._match_counts(pred_boxes, pred_scores, gt_boxes, thr, iou_threshold)
                        rows[(class_id, thr)]["tp"] += tp
                        rows[(class_id, thr)]["fp"] += fp
                        rows[(class_id, thr)]["fn"] += fn
                        rows[(class_id, thr)]["pred"] += tp + fp
                        rows[(class_id, thr)]["gt"] += int(gt_boxes.shape[0])

        class_rows = []
        for class_id in range(num_classes):
            best_class = None
            for thr in thresholds:
                row = rows[(class_id, thr)]
                precision, recall, f1 = self._score_from_counts(row["tp"], row["fp"], row["fn"])
                row.update({"precision": precision, "recall": recall, "f1": f1})
                if best_class is None or f1 > best_class["f1"]:
                    best_class = dict(row)
            class_rows.append(best_class)

        sweep_rows = []
        best_global = None
        for thr in thresholds:
            threshold_rows = [rows[(class_id, thr)] for class_id in range(num_classes)]
            tp = sum(row["tp"] for row in threshold_rows)
            fp = sum(row["fp"] for row in threshold_rows)
            fn = sum(row["fn"] for row in threshold_rows)
            micro_precision, micro_recall, micro_f1 = self._score_from_counts(tp, fp, fn)
            macro_precision = sum(row["precision"] for row in threshold_rows) / max(num_classes, 1)
            macro_recall = sum(row["recall"] for row in threshold_rows) / max(num_classes, 1)
            macro_f1 = sum(row["f1"] for row in threshold_rows) / max(num_classes, 1)
            global_row = {
                "class_id": -1,
                "class_name": "all_macro_micro",
                "threshold": thr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pred": tp + fp,
                "gt": tp + fn,
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }
            sweep_rows.append(global_row)
            sweep_rows.extend(rows[(class_id, thr)] for class_id in range(num_classes))
            if best_global is None or macro_f1 > best_global["f1"]:
                best_global = global_row

        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "val_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "class_id", "class_name", "threshold", "tp", "fp", "fn", "pred", "gt",
                    "precision", "recall", "f1", "micro_precision", "micro_recall", "micro_f1",
                ])
                writer.writeheader()
                writer.writerows(sweep_rows)
        return {
            "best_global": best_global or {
                "threshold": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
            },
            "per_class": class_rows,
        }

    def _evaluate_teacher(self, trainer: SSODTrainer, epoch: int):
        test_stats, _ = evaluate(
            trainer.teacher,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )
        sweep = self._threshold_sweep(trainer.teacher, iou_threshold=0.5)
        best = sweep["best_global"]
        per_class = sweep["per_class"]
        stats = {
            "epoch": epoch,
            "best_threshold": float(best["threshold"]),
            "macro_precision_iou50": float(best["precision"]),
            "macro_recall_iou50": float(best["recall"]),
            "macro_f1_iou50": float(best["f1"]),
            "micro_precision_iou50": float(best["micro_precision"]),
            "micro_recall_iou50": float(best["micro_recall"]),
            "micro_f1_iou50": float(best["micro_f1"]),
            "per_class_iou50": {
                str(row["class_id"]): {
                    "name": row["class_name"],
                    "best_threshold": float(row["threshold"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                }
                for row in per_class
            },
        }
        if "coco_eval_bbox" in test_stats:
            stats["coco_bbox_mAP"] = float(test_stats["coco_eval_bbox"][0])
            stats["coco_bbox_mAP50"] = float(test_stats["coco_eval_bbox"][1])
        print(f"Teacher EMA val epoch={epoch}: {stats}")
        return stats

    def _ssod_state_dict(self, trainer: SSODTrainer, epoch: int, train_stats=None, val_stats=None):
        state = self.state_dict()
        state["last_epoch"] = epoch
        state["teacher"] = dist_utils.de_parallel(trainer.teacher).state_dict()
        state["ssod_train_stats"] = train_stats or {}
        state["ssod_val_stats"] = val_stats or {}
        if getattr(trainer, "adaptive_threshold", None) is not None:
            # Persist per-class score history + cached thresholds so a resume
            # does not have to spend ~50 steps rebuilding adaptive thresholds
            # from zero (which would silently fall back to ``pseudo_threshold``
            # for the first SSOD steps after resume).
            state["adaptive_threshold"] = trainer.adaptive_threshold.state_dict()
        return state

    def val(self):
        raise NotImplementedError("Use the normal detection solver for evaluation checkpoints.")
