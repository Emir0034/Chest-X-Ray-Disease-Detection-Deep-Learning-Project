import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers and scripts
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filepath: str) -> None:
    """Save a training checkpoint to disk.

    Expected keys in state:
        epoch, model_state_dict, optimizer_state_dict, val_auc, exp_name
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """Load a checkpoint into model (and optionally optimizer).

    Returns the full checkpoint dict so callers can recover epoch / val_auc.
    """
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def plot_training_curves(history: dict, save_path: str) -> None:
    """Save a 2-panel training curve figure (loss | val AUC).

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_auc'
                 each mapping to a list of per-epoch float values.
        save_path: full path (including filename) where the PNG is saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="coral")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_auc"], label="Val Avg AUC", color="seagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("Validation Average AUC")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_auc_per_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list,
) -> dict:
    """Compute per-class AUC-ROC scores.

    Returns a dict mapping each label name to its AUC (or float('nan') if
    the class has only one unique ground-truth value in the evaluated set,
    which makes AUC undefined).
    """
    auc_dict = {}
    for i, name in enumerate(label_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_score[:, i])
        except ValueError:
            auc = float("nan")
        auc_dict[name] = auc
    return auc_dict
