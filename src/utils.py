"""Утилиты: воспроизводимость, метры, ранняя остановка."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Зафиксировать seed для воспроизводимости.

    deterministic=True гарантирует битовую идентичность между запусками,
    ценой ~10% скорости (cudnn.benchmark=False).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """Скользящее среднее (для loss/accuracy в эпохе)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@dataclass
class EarlyStopping:
    """Останавливает обучение, если метрика не улучшается patience эпох.

    mode="max" — для accuracy/F1; mode="min" — для loss.
    """

    patience: int = 7
    mode: str = "max"
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        self.best: float = -float("inf") if self.mode == "max" else float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_epoch: int = -1

    def step(self, value: float, epoch: int) -> bool:
        """True, если метрика улучшилась."""
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def human_format(num: float) -> str:
    """1234567 -> '1.23M'."""
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}T"


class Timer:
    """Контекст-таймер: with Timer() as t: ...; print(t.elapsed)."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
