"""Two-stage fit loop с дифференцированным lr, AMP, EarlyStopping, TB logging."""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import NUM_CLASSES
from .configs import TrainConfig
from .dataset import GTZANDataset, mixup_batch
from .eval import aggregate_window_predictions, compute_metrics
from .utils import AverageMeter, EarlyStopping, Timer, ensure_dir, set_seed


# ============================================================
# Loss с поддержкой soft-targets (mixup)
# ============================================================
def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """CrossEntropy с soft labels (для mixup).

    Если label_smoothing > 0 — применяется поверх soft_targets:
        target = (1 - eps) * soft + eps / C
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        eps = label_smoothing
        soft_targets = (1.0 - eps) * soft_targets + eps / soft_targets.size(-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
) -> dict:
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    use_mixup = cfg.aug.use_mixup and cfg.aug.mixup_alpha > 0

    for mel, label, _ in loader:
        mel = mel.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if use_mixup:
            mixed_x, soft_y = mixup_batch(mel, label, cfg.aug.mixup_alpha, NUM_CLASSES)
        else:
            mixed_x = mel
            soft_y = torch.zeros(label.size(0), NUM_CLASSES, device=device)
            soft_y.scatter_(1, label.unsqueeze(1), 1.0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and cfg.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(mixed_x)
                loss = soft_cross_entropy(out["logits"], soft_y, cfg.label_smoothing)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(mixed_x)
            loss = soft_cross_entropy(out["logits"], soft_y, cfg.label_smoothing)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        loss_m.update(loss.item(), mel.size(0))
        # Accuracy по hard label (для отслеживания)
        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            acc_m.update((pred == label).float().mean().item(), mel.size(0))

    return {"train_loss": loss_m.avg, "train_acc": acc_m.avg}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict:
    """Валидация с агрегацией по окнам трека."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_groups: list[int] = []
    loss_m = AverageMeter()
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    for mel, label, group in loader:
        mel = mel.to(device, non_blocking=True)
        label_dev = label.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=cfg.use_amp):
            out = model(mel)
            loss = ce(out["logits"], label_dev)
        probs = F.softmax(out["logits"].float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(label.tolist())
        all_groups.extend(group.tolist())
        loss_m.update(loss.item(), mel.size(0))

    all_probs_np = np.concatenate(all_probs, axis=0)
    track_probs, track_labels = aggregate_window_predictions(
        all_probs_np, np.array(all_labels), np.array(all_groups), method="median"
    )
    track_preds = track_probs.argmax(axis=-1)
    metrics = compute_metrics(track_labels, track_preds, track_probs)
    metrics["val_loss"] = loss_m.avg
    return {**metrics, "_track_probs": track_probs, "_track_labels": track_labels}


# ============================================================
# Schedulers: warmup + cosine
# ============================================================
def make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup + cosine decay."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Main two-stage fit
# ============================================================
def fit_two_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    out_dir: Path | str,
    tb_dir: Path | str | None = None,
) -> dict:
    """Полный цикл обучения с двумя стадиями.

    Возвращает dict с историей метрик и пути к best checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(out_dir)
    set_seed(cfg.seed, deterministic=True)
    model = model.to(device)
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir else None
    history: list[dict] = []
    best_path = out_dir / f"best_{cfg.model_name}_fold{cfg.fold}_seed{cfg.seed}.pth"

    # ======= STAGE 1: head only =======
    if not cfg.full_ft_from_start:
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr_head, weight_decay=cfg.wd_head,
        )
        scheduler = make_scheduler(optimizer, cfg.stage1_epochs, warmup_epochs=0)
        scaler = torch.cuda.amp.GradScaler() if cfg.use_amp else None
        es = EarlyStopping(patience=cfg.early_stop_patience, mode="max")

        for epoch in range(cfg.stage1_epochs):
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
            val = validate(model, val_loader, device, cfg)
            scheduler.step()
            row = {
                "stage": 1, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - t0, **tr,
                **{k: v for k, v in val.items() if not k.startswith("_")},
            }
            history.append(row)
            print(f"[stage1 e{epoch:02d}] loss={tr['train_loss']:.3f} "
                  f"train_acc={tr['train_acc']:.3f} val_acc={val['accuracy']:.3f} "
                  f"val_f1={val['macro_f1']:.3f} ({row['elapsed_s']:.1f}s)")
            if writer:
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"stage1/{k}", v, epoch)
            improved = es.step(val["macro_f1"], epoch)
            if improved:
                torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch,
                            "val_f1": val["macro_f1"], "stage": 1}, best_path)

    # ======= STAGE 2: unfreeze last block (или all if full_ft) =======
    if cfg.full_ft_from_start:
        model.unfreeze_all()
    else:
        model.unfreeze_last_block()
    # Дифференцированный lr через param_groups
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.wd_backbone, cfg.wd_head)
    )
    scheduler = make_scheduler(optimizer, cfg.stage2_epochs, warmup_epochs=cfg.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler() if cfg.use_amp else None
    es = EarlyStopping(patience=cfg.early_stop_patience, mode="max")
    # Загружаем best из stage 1, если есть
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        es.best = ckpt.get("val_f1", -float("inf"))
        print(f"[stage2] resume from stage1 best val_f1={es.best:.3f}")

    for epoch in range(cfg.stage2_epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val = validate(model, val_loader, device, cfg)
        scheduler.step()
        row = {
            "stage": 2, "epoch": epoch,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "elapsed_s": time.time() - t0, **tr,
            **{k: v for k, v in val.items() if not k.startswith("_")},
        }
        history.append(row)
        print(f"[stage2 e{epoch:02d}] loss={tr['train_loss']:.3f} "
              f"train_acc={tr['train_acc']:.3f} val_acc={val['accuracy']:.3f} "
              f"val_f1={val['macro_f1']:.3f} ({row['elapsed_s']:.1f}s)")
        if writer:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"stage2/{k}", v, epoch)
        improved = es.step(val["macro_f1"], epoch)
        if improved:
            torch.save({
                "state_dict": model.state_dict(), "cfg": asdict(cfg),
                "epoch": epoch, "val_f1": val["macro_f1"], "stage": 2,
                "history": history,
            }, best_path)
        if es.should_stop:
            print(f"[early stop] no improvement for {cfg.early_stop_patience} epochs")
            break

    if writer:
        writer.close()
    return {
        "history": history,
        "best_path": str(best_path),
        "best_val_f1": es.best,
        "best_epoch": es.best_epoch,
    }


# ============================================================
# Helpers
# ============================================================
def build_loaders(
    h5_path: str,
    train_ids: list[str],
    val_ids: list[str],
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_ds = GTZANDataset(h5_path, train_ids, mode="train", aug_cfg=cfg.aug)
    val_ds = GTZANDataset(h5_path, val_ids, mode="eval", aug_cfg=cfg.aug)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader
