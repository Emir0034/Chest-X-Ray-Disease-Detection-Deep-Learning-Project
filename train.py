import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
from dataset import create_patient_splits, get_dataloaders
from loss import get_loss_fn
from model import build_model
from utils import (
    compute_auc_per_class,
    load_checkpoint,
    plot_training_curves,
    save_checkpoint,
    seed_everything,
)


# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    """Run one training epoch with mixed-precision AMP.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label_names: list,
) -> tuple:
    """Run validation and return loss, average AUC, and per-class AUC.

    Returns:
        (val_loss, avg_auc, per_class_auc_dict)
    """
    model.eval()
    all_logits  = []
    all_labels  = []
    total_loss  = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = loss_fn(logits, labels)
            total_loss  += loss.item()
            all_logits.append(logits.cpu().float())
            all_labels.append(labels.cpu().float())

    val_loss     = total_loss / len(loader)
    all_logits   = torch.cat(all_logits, dim=0)
    all_labels   = torch.cat(all_labels, dim=0)
    all_probs    = torch.sigmoid(all_logits).numpy()
    all_labels_np = all_labels.numpy()

    per_class_auc = compute_auc_per_class(all_labels_np, all_probs, label_names)
    valid_aucs    = [v for v in per_class_auc.values() if not np.isnan(v)]
    avg_auc       = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    return val_loss, avg_auc, per_class_auc


def main() -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    seed_everything(config.SEED)

    for d in [config.CHECKPOINTS_DIR, config.METRICS_DIR, config.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    exp_name = config.get_experiment_name()
    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Device     : {DEVICE}")
    print(f"CLAHE      : {config.USE_CLAHE}")
    print(f"CBAM       : {config.USE_CBAM}")
    print(f"Loss       : {'AsymmetricLoss' if config.USE_ASYMMETRIC_LOSS else 'BCEWithLogitsLoss'}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = create_patient_splits(
        csv_path   = config.DATA_CSV,
        split_dir  = config.SPLIT_DIR,
        seed       = config.SEED,
    )
    train_loader, val_loader, _ = get_dataloaders(
        train_df    = train_df,
        val_df      = val_df,
        test_df     = test_df,
        images_dir  = config.IMAGES_DIR,
        use_clahe   = config.USE_CLAHE,
        batch_size  = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
    )

    # ── Model / optimiser / scheduler ────────────────────────────────────────
    model     = build_model(config).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )
    loss_fn = get_loss_fn(config.USE_ASYMMETRIC_LOSS)
    scaler  = torch.cuda.amp.GradScaler()

    # ── Training state ────────────────────────────────────────────────────────
    best_val_auc     = 0.0
    best_epoch       = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_auc":    [],
    }
    # per-class AUC history: list of dicts, one per epoch
    per_class_auc_history = []

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
        val_loss, avg_auc, per_class_auc = validate(model, val_loader, loss_fn, DEVICE, config.DISEASE_LABELS)

        scheduler.step(avg_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(avg_auc)
        per_class_auc_history.append(per_class_auc)

        # ── Console logging ───────────────────────────────────────────────────
        print(
            f"Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Avg AUC: {avg_auc:.4f}"
        )
        # Print per-class AUC table every 5 epochs and on the final epoch
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            print(f"\n  {'Class':<22} {'Val AUC':>8}")
            print(f"  {'-'*32}")
            for cls, auc in per_class_auc.items():
                auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
                print(f"  {cls:<22} {auc_str:>8}")
            print()

        # ── Checkpoint saving ─────────────────────────────────────────────────
        last_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f"{exp_name}_last.pth")
        save_checkpoint(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc":              avg_auc,
                "exp_name":             exp_name,
            },
            last_ckpt_path,
        )

        if avg_auc > best_val_auc:
            best_val_auc     = avg_auc
            best_epoch       = epoch
            patience_counter = 0
            best_ckpt_path   = os.path.join(config.CHECKPOINTS_DIR, f"{exp_name}_best.pth")
            save_checkpoint(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc":              avg_auc,
                    "exp_name":             exp_name,
                },
                best_ckpt_path,
            )
            print(f"  ✓ New best checkpoint saved (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # ── Save all training outputs ──────────────────────────────────────────────

    # 1. Training curves PNG
    curves_path = os.path.join(config.FIGURES_DIR, f"{exp_name}_curves.png")
    plot_training_curves(history, curves_path)
    print(f"\nTraining curves saved: {curves_path}")

    # 2. Train/val loss history CSV
    loss_csv_path = os.path.join(config.METRICS_DIR, f"{exp_name}_loss_history.csv")
    loss_df = pd.DataFrame(
        {
            "epoch":      range(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss":   history["val_loss"],
        }
    )
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Loss history CSV saved: {loss_csv_path}")

    # 3. Per-epoch validation AUC history CSV (avg + per-class)
    auc_rows = []
    for i, (avg, pc) in enumerate(zip(history["val_auc"], per_class_auc_history), start=1):
        row = {"epoch": i, "avg_auc": avg}
        row.update(pc)
        auc_rows.append(row)
    auc_csv_path = os.path.join(config.METRICS_DIR, f"{exp_name}_val_auc_history.csv")
    pd.DataFrame(auc_rows).to_csv(auc_csv_path, index=False)
    print(f"Val AUC history CSV saved: {auc_csv_path}")

    # 4. Best epoch summary JSON
    best_epoch_path = os.path.join(config.METRICS_DIR, f"{exp_name}_best_epoch.json")
    with open(best_epoch_path, "w") as f:
        json.dump(
            {"exp_name": exp_name, "best_epoch": best_epoch, "best_val_auc": best_val_auc},
            f, indent=2,
        )
    print(f"Best epoch JSON saved: {best_epoch_path}")

    print(f"\nTraining complete.")
    print(f"Best epoch : {best_epoch}  |  Best Val AUC : {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
