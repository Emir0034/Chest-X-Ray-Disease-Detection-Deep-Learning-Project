import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config
from dataset import create_patient_splits, get_dataloaders
from model import build_model
from utils import compute_auc_per_class, load_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    label_names: list,
    threshold: float = 0.5,
) -> dict:
    """Run inference on a DataLoader and compute all test metrics.

    Args:
        model:       Trained ChestXRayModel in eval mode.
        loader:      Test DataLoader.
        device:      Compute device.
        label_names: List of 14 disease class names.
        threshold:   Decision threshold for Precision / Recall / F1 (default 0.5).

    Returns:
        metrics dict with keys:
            avg_auc, per_class_auc, precision, recall, f1, threshold
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.sigmoid(logits).cpu().float()
            all_probs.append(probs)
            all_labels.append(labels.float())

    y_score = torch.cat(all_probs,  dim=0).numpy()   # (N, 14)
    y_true  = torch.cat(all_labels, dim=0).numpy()   # (N, 14)
    y_pred  = (y_score >= threshold).astype(int)

    per_class_auc = compute_auc_per_class(y_true, y_score, label_names)
    valid_aucs    = [v for v in per_class_auc.values() if not np.isnan(v)]
    avg_auc       = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_true,    y_pred, average="macro", zero_division=0))
    f1        = float(f1_score(y_true,        y_pred, average="macro", zero_division=0))

    return {
        "avg_auc":      avg_auc,
        "per_class_auc": per_class_auc,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "threshold":    threshold,
    }


def save_auc_bar_chart(per_class_auc: dict, avg_auc: float, save_path: str, exp_name: str) -> None:
    """Save a horizontal bar chart of per-class AUC scores.

    Classes with undefined AUC (NaN) are shown with a grey bar and labelled N/A.
    A vertical dashed line marks the average AUC.
    """
    classes = list(per_class_auc.keys())
    aucs    = [per_class_auc[c] for c in classes]

    # Replace NaN with 0 for plotting; track which are undefined
    plot_aucs = [a if not np.isnan(a) else 0.0 for a in aucs]
    colors    = ["steelblue" if not np.isnan(a) else "lightgrey" for a in aucs]

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(classes, plot_aucs, color=colors, edgecolor="white", height=0.6)

    # Annotate bar values
    for bar, auc in zip(bars, aucs):
        label = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8,
        )

    # Average AUC reference line
    ax.axvline(avg_auc, color="coral", linestyle="--", linewidth=1.5, label=f"Avg AUC = {avg_auc:.3f}")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(f"Per-Class Test AUC — {exp_name}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    exp_name = config.get_experiment_name()
    print(f"\n{'='*60}")
    print(f"Evaluating : {exp_name}")
    print(f"Device     : {DEVICE}")
    print(f"{'='*60}\n")

    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # ── Load test split ───────────────────────────────────────────────────────
    _, _, test_df = create_patient_splits(
        csv_path  = config.DATA_CSV,
        split_dir = config.SPLIT_DIR,
        seed      = config.SEED,
    )
    _, _, test_loader = get_dataloaders(
        train_df    = test_df,   # dummy — not used
        val_df      = test_df,   # dummy — not used
        test_df     = test_df,
        images_dir  = config.IMAGES_DIR,
        use_clahe   = config.USE_CLAHE,
        batch_size  = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
    )

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_ckpt = os.path.join(config.CHECKPOINTS_DIR, f"{exp_name}_best.pth")
    model = build_model(config).to(DEVICE)
    ckpt  = load_checkpoint(best_ckpt, model)
    print(f"Loaded checkpoint: {best_ckpt}")
    print(f"  (trained to epoch {ckpt.get('epoch', '?')}, val AUC {ckpt.get('val_auc', '?'):.4f})\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, test_loader, DEVICE, config.DISEASE_LABELS)
    metrics["exp_name"] = exp_name

    # ── Print results table ───────────────────────────────────────────────────
    print(f"  {'Class':<22} {'Test AUC':>10}")
    print(f"  {'-'*36}")
    for cls, auc in metrics["per_class_auc"].items():
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A  "
        print(f"  {cls:<22} {auc_str:>10}")
    print(f"  {'-'*36}")
    print(f"  {'Average AUC':<22} {metrics['avg_auc']:>10.4f}")
    print(f"\n  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  Threshold : {metrics['threshold']}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────

    # 1. Per-class AUC CSV
    auc_csv_path = os.path.join(config.METRICS_DIR, f"{exp_name}_test_auc_per_class.csv")
    auc_df = pd.DataFrame(
        [{"class": cls, "auc": auc} for cls, auc in metrics["per_class_auc"].items()]
    )
    auc_df.to_csv(auc_csv_path, index=False)
    print(f"Per-class AUC CSV saved : {auc_csv_path}")

    # 2. Test summary metrics CSV
    summary_csv_path = os.path.join(config.METRICS_DIR, f"{exp_name}_test_summary.csv")
    pd.DataFrame([{
        "exp_name":  exp_name,
        "avg_auc":   metrics["avg_auc"],
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "f1":        metrics["f1"],
        "threshold": metrics["threshold"],
    }]).to_csv(summary_csv_path, index=False)
    print(f"Test summary CSV saved  : {summary_csv_path}")

    # 3. Full metrics JSON
    json_path = os.path.join(config.METRICS_DIR, f"{exp_name}_test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Full metrics JSON saved : {json_path}")

    # 4. Per-class AUC bar chart PNG
    bar_chart_path = os.path.join(config.FIGURES_DIR, f"{exp_name}_test_auc_bar.png")
    save_auc_bar_chart(metrics["per_class_auc"], metrics["avg_auc"], bar_chart_path, exp_name)
    print(f"AUC bar chart PNG saved : {bar_chart_path}")


if __name__ == "__main__":
    main()
