"""Визуализации: curves, confusion matrix, t-SNE/UMAP, Grad-CAM, reliability.

Все функции принимают данные из metrics csv / numpy и возвращают matplotlib Figure
для сохранения в PNG/PDF из ноутбука.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import GENRES

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


# ============================================================
# 1. Training curves: loss & accuracy с std-band по фолдам
# ============================================================
def plot_training_curves(
    history_df: pd.DataFrame,   # cols: model, fold, seed, epoch_global, train_loss, val_loss, val_acc, val_f1
    out_path: Path | str | None = None,
) -> plt.Figure:
    """4-panel: train_loss / val_loss / val_acc / val_f1 с mean ± std по (fold, seed)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_panels = [
        ("train_loss", "Train loss", axes[0, 0]),
        ("val_loss",   "Val loss",   axes[0, 1]),
        ("val_acc",    "Val accuracy", axes[1, 0]),
        ("val_f1",     "Val macro-F1", axes[1, 1]),
    ]
    palette = {"cnn14": "#1f77b4", "resnet18": "#ff7f0e"}

    for metric, title, ax in metric_panels:
        if metric not in history_df.columns:
            continue
        for model_name, group in history_df.groupby("model"):
            agg = group.groupby("epoch_global")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(agg["epoch_global"], agg["mean"], label=model_name, color=palette.get(model_name, "gray"))
            ax.fill_between(
                agg["epoch_global"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                alpha=0.2, color=palette.get(model_name, "gray"),
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle("Training dynamics (mean ± std across 5 folds × 3 seeds)", fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 2. Confusion matrices side-by-side
# ============================================================
def plot_confusion_matrices(
    cms: dict[str, np.ndarray],   # {"cnn14": (10,10), "resnet18": (10,10)}
    normalize: bool = True,
    out_path: Path | str | None = None,
) -> plt.Figure:
    n_models = len(cms)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        cm_arr = np.array(cm).astype(float)
        if normalize:
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_arr = cm_arr / row_sums
        sns.heatmap(
            cm_arr, annot=True, fmt=".2f" if normalize else "d",
            cmap="Blues", cbar=True, square=True,
            xticklabels=list(GENRES), yticklabels=list(GENRES), ax=ax,
            vmin=0, vmax=1 if normalize else None,
        )
        ax.set_title(f"{name} (acc-diag={np.trace(cm_arr)/cm_arr.sum():.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion matrix" + (" (row-normalized)" if normalize else ""), fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 3. Per-class F1 bar chart
# ============================================================
def plot_per_class_f1(
    per_class_f1: dict[str, list[float]],  # {"cnn14": [...], "resnet18": [...]}
    out_path: Path | str | None = None,
) -> plt.Figure:
    df = pd.DataFrame(per_class_f1, index=list(GENRES))
    df = df.reset_index().rename(columns={"index": "genre"}).melt(
        id_vars="genre", var_name="model", value_name="f1",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df, x="genre", y="f1", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 score")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 4. t-SNE / UMAP embeddings
# ============================================================
def plot_embeddings_2d(
    embeddings: dict[str, np.ndarray],     # {"cnn14": (N, D), "resnet18": (N, D)}
    labels: np.ndarray,                    # (N,) — общие labels для обоих
    method: str = "umap",
    random_state: int = 42,
    out_path: Path | str | None = None,
) -> plt.Figure:
    n_models = len(embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    palette = sns.color_palette("tab10", n_colors=len(GENRES))

    for ax, (name, emb) in zip(axes, embeddings.items()):
        if method == "umap":
            try:
                import umap  # type: ignore
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
                emb2d = reducer.fit_transform(emb)
            except Exception:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, perplexity=30, random_state=random_state, init="pca")
                emb2d = reducer.fit_transform(emb)
                method = "tsne_fallback"
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=30, random_state=random_state, init="pca")
            emb2d = reducer.fit_transform(emb)
        for cls_idx, genre in enumerate(GENRES):
            m = labels == cls_idx
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=12, alpha=0.7, label=genre, color=palette[cls_idx])
        ax.set_title(f"{name} ({method.upper()})")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 5. Calibration: reliability diagram
# ============================================================
def plot_reliability(
    ece_info: dict[str, dict],  # {"cnn14": {bin_accs, bin_confs, bin_counts}, ...}
    eces: dict[str, float],
    out_path: Path | str | None = None,
) -> plt.Figure:
    n_models = len(ece_info)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, info) in zip(axes, ece_info.items()):
        bin_confs = info["bin_confs"]
        bin_accs = info["bin_accs"]
        bin_counts = np.array(info["bin_counts"], dtype=float)
        width = 1.0 / len(bin_confs) * 0.9
        ax.bar(bin_confs, bin_accs, width=width, edgecolor="black",
               alpha=0.8, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} (ECE={eces.get(name, 0.0):.3f})")
        ax.legend(loc="upper left")
    fig.suptitle("Reliability diagram", fontsize=14)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 6. Grad-CAM
# ============================================================
def compute_gradcam(
    model,
    target_layer,
    x: "torch.Tensor",            # noqa: F821 (lazy import torch)
    target_class: int | None = None,
) -> np.ndarray:
    """Простая Grad-CAM реализация без зависимости от pytorch-grad-cam.

    Возвращает heatmap shape как H×W входной spatial-features.
    x: (1, 1, n_mels, T).
    """
    import torch
    model.eval()
    activations: list = []
    gradients: list = []

    def fwd_hook(m, inp, out):
        activations.append(out.detach())

    def bwd_hook(m, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()
        act = activations[0][0]      # (C, H, W)
        grad = gradients[0][0]       # (C, H, W)
        weights = grad.mean(dim=(1, 2))         # (C,)
        cam = (weights[:, None, None] * act).sum(dim=0)
        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def plot_gradcam_grid(
    mels: dict[str, np.ndarray],          # {"cnn14": (H, W), "resnet18": (H, W)}
    cams: dict[str, np.ndarray],          # same keys, same shapes (or upsampled)
    title: str,
    out_path: Path | str | None = None,
) -> plt.Figure:
    """Сравнение Grad-CAM двух моделей на одном треке."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (name, mel) in zip(axes, mels.items()):
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        cam = cams[name]
        # upsample cam до размера mel
        from scipy.ndimage import zoom
        zoom_y = mel.shape[0] / cam.shape[0]
        zoom_x = mel.shape[1] / cam.shape[1]
        cam_up = zoom(cam, (zoom_y, zoom_x), order=1)
        ax.imshow(cam_up, aspect="auto", origin="lower", cmap="jet", alpha=0.45)
        ax.set_title(f"{name}")
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel bin")
    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 7. Bar chart: training time × accuracy (Pareto)
# ============================================================
def plot_cost_vs_accuracy(
    summary: pd.DataFrame,   # cols: model, mean_acc, std_acc, train_time_h, params_m
    out_path: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"cnn14": "#1f77b4", "resnet18": "#ff7f0e"}
    for _, row in summary.iterrows():
        ax.errorbar(row["train_time_h"], row["mean_acc"],
                    yerr=row["std_acc"], fmt="o", markersize=row["params_m"] * 0.3 + 5,
                    color=palette.get(row["model"], "gray"),
                    capsize=4, label=f"{row['model']} ({row['params_m']:.1f}M)")
        ax.annotate(row["model"], xy=(row["train_time_h"], row["mean_acc"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Training time, hours (15 runs total)")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Accuracy vs training cost (marker size ∝ params)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# 8. Time-to-90% bar chart
# ============================================================
def plot_time_to_fraction(
    ttf_data: dict[str, list[int]],   # {"cnn14": [epoch_for_each_run], "resnet18": [...]}
    fraction: float = 0.9,
    out_path: Path | str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    for name, vals in ttf_data.items():
        for v in vals:
            data.append({"model": name, "epoch": v})
    df = pd.DataFrame(data)
    sns.boxplot(data=df, x="model", y="epoch", ax=ax)
    sns.stripplot(data=df, x="model", y="epoch", ax=ax, color="black", size=4, alpha=0.7)
    ax.set_title(f"Time to {int(fraction*100)}% of best val-F1 (epochs across runs)")
    ax.set_ylabel("Epoch")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
