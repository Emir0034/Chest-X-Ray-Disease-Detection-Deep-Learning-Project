import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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


def _compute_faar_freq_weights(train_df, class_names, mode="inverse", cap_max=3.0):
    """Compute normalized frequency weights for FAAR from the training split.

    Supports both one-hot column layout and NIH pipe-separated 'Finding Labels'.

    Modes:
        "inverse"  — 1/(freq+eps), normalized by mean
        "cap"      — inverse, normalized, capped at cap_max, re-normalized
        "sqrt_inv" — 1/sqrt(freq+eps), normalized by mean
    """
    n = len(train_df)
    if all(c in train_df.columns for c in class_names):
        pos = train_df[class_names].values.sum(axis=0).astype(float)
    elif "Finding Labels" in train_df.columns:
        pos = np.array(
            [train_df["Finding Labels"].str.contains(cls, regex=False).sum()
             for cls in class_names],
            dtype=float,
        )
    else:
        raise ValueError(
            "_compute_faar_freq_weights: train_df has neither per-class columns "
            "nor a 'Finding Labels' column."
        )
    freq = pos / n
    if mode == "inverse":
        w = 1.0 / (freq + 1e-6)
        w = w / w.mean()
    elif mode == "cap":
        w = 1.0 / (freq + 1e-6)
        w = w / w.mean()
        w = np.minimum(w, cap_max)
        w = w / w.mean()
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(freq + 1e-6)
        w = w / w.mean()
    else:
        raise ValueError(f"_compute_faar_freq_weights: unknown mode {mode!r}")
    return torch.tensor(w, dtype=torch.float32)


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

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="  Train", leave=False, unit="batch", dynamic_ncols=True, position=1)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        if loss_val != loss_val or abs(loss_val) == float("inf"):
            raise RuntimeError(
                f"NaN or Inf loss at batch {batch_idx} (loss={loss_val}). "
                "Check loss function, input data, or learning rate."
            )
        total_loss += loss_val

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
        for images, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch", dynamic_ncols=True, position=1):
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
    print(f"Status     : {config.EXPERIMENT_STATUS}")
    print(f"Device     : {DEVICE}")
    print(f"Checkpoints: {config.CHECKPOINTS_DIR}")
    print(f"Metrics    : {config.METRICS_DIR}")
    print(f"Figures    : {config.FIGURES_DIR}")
    print(f"CLAHE      : {config.USE_CLAHE}")
    print(f"CBAM       : {config.USE_CBAM}")
    if config.USE_ASYMMETRIC_LOSS:
        print(f"Loss       : AsymmetricLoss")
        print(f"ASL params : gamma_neg={config.ASL_GAMMA_NEG}, gamma_pos={config.ASL_GAMMA_POS}, clip={config.ASL_CLIP}")
    elif getattr(config, 'USE_FOCAL_LOSS', False):
        print(f"Loss       : FocalLoss")
        print(f"Focal params : gamma={config.FOCAL_GAMMA}, alpha={config.FOCAL_ALPHA}")
    else:
        print(f"Loss       : BCEWithLogitsLoss")
    print(f"FAAR       : {'enabled' if getattr(config, 'USE_FAAR', False) else 'disabled'}")
    if getattr(config, "USE_FAAR", False):
        print(f"FAAR mode  : {getattr(config, 'FAAR_WEIGHT_MODE', 'inverse')}")
        if getattr(config, "FAAR_WEIGHT_MODE", "inverse") == "cap":
            print(f"FAAR cap   : {getattr(config, 'FAAR_CAP_MAX', 3.0)}")
        print(f"FAAR alpha : init={getattr(config, 'FAAR_ALPHA_INIT', 0.0)}")
    print(f"Optimizer  : {getattr(config, 'OPTIMIZER_NAME', 'adam').upper()}")
    print(f"Weight Decay: {getattr(config, 'WEIGHT_DECAY', 1e-5)}")
    _sched_name = getattr(config, "SCHEDULER_NAME", "plateau")
    print(f"Scheduler  : {_sched_name}")
    if _sched_name == "cosine":
        print(f"  T_max    : {getattr(config, 'COSINE_T_MAX', 10)}")
        print(f"  eta_min  : {getattr(config, 'COSINE_ETA_MIN', 1e-6)}")
    _ft_display = getattr(config, "FINE_TUNE_FROM_CHECKPOINT", "") or "no"
    print(f"Fine-tune  : {_ft_display}")
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
    if getattr(config, "USE_FAAR", False):
        fw = _compute_faar_freq_weights(
            train_df,
            config.DISEASE_LABELS,
            mode=getattr(config, "FAAR_WEIGHT_MODE", "inverse"),
            cap_max=getattr(config, "FAAR_CAP_MAX", 3.0),
        )
        print(f"FAAR freq_weights : {fw.numpy().round(4).tolist()}\n")
        model = build_model(config, faar_freq_weights=fw).to(DEVICE)
    else:
        model = build_model(config).to(DEVICE)

    # ── Fine-tune initialization (model weights only) ─────────────────────────
    _ft_ckpt = getattr(config, "FINE_TUNE_FROM_CHECKPOINT", "")
    if _ft_ckpt:
        if not os.path.isfile(_ft_ckpt):
            raise FileNotFoundError(
                f"FINE_TUNE_FROM_CHECKPOINT: checkpoint not found: {_ft_ckpt!r}"
            )
        print(f"Fine-tuning initialized from checkpoint: {_ft_ckpt}")
        _ft_raw = torch.load(_ft_ckpt, map_location=DEVICE)
        if isinstance(_ft_raw, dict) and "model_state_dict" in _ft_raw:
            _ft_state = _ft_raw["model_state_dict"]
        elif isinstance(_ft_raw, dict) and "state_dict" in _ft_raw:
            _ft_state = _ft_raw["state_dict"]
        else:
            _ft_state = _ft_raw
        if any(k.startswith("module.") for k in _ft_state):
            _ft_state = {k[len("module."):]: v for k, v in _ft_state.items()}
        model.load_state_dict(_ft_state, strict=True)
        print("Starting a fresh fine-tuning run from epoch 1.")
    else:
        print("Starting training from scratch / ImageNet-pretrained initialization.")

    _opt_name = getattr(config, "OPTIMIZER_NAME", "adam").lower()
    _wd       = getattr(config, "WEIGHT_DECAY", 1e-5)
    if _opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=_wd,
        )
    elif _opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=_wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER_NAME!r}")
    _sched = getattr(config, "SCHEDULER_NAME", "plateau")
    if _sched == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
    elif _sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(config, "COSINE_T_MAX", 10),
            eta_min=getattr(config, "COSINE_ETA_MIN", 1e-6),
        )
    else:
        raise ValueError(
            f"Unknown SCHEDULER_NAME: {_sched!r}. Must be 'plateau' or 'cosine'."
        )
    loss_fn = get_loss_fn(
        use_asymmetric=config.USE_ASYMMETRIC_LOSS,
        gamma_neg=config.ASL_GAMMA_NEG,
        gamma_pos=config.ASL_GAMMA_POS,
        clip=config.ASL_CLIP,
        use_focal=getattr(config, 'USE_FOCAL_LOSS', False),
        focal_gamma=getattr(config, 'FOCAL_GAMMA', 2.0),
        focal_alpha=getattr(config, 'FOCAL_ALPHA', None),
    )
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
    epoch_bar = tqdm(
        range(1, config.NUM_EPOCHS + 1),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
        position=0,
    )
    for epoch in epoch_bar:
        _cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1}  LR: {_cur_lr:.2e}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
        val_loss, avg_auc, per_class_auc = validate(model, val_loader, loss_fn, DEVICE, config.DISEASE_LABELS)

        if getattr(config, "SCHEDULER_NAME", "plateau") == "cosine":
            scheduler.step()
        else:
            scheduler.step(avg_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(avg_auc)
        per_class_auc_history.append(per_class_auc)

        # ── Checkpoint saving ─────────────────────────────────────────────────
        ckpt_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc":              avg_auc,
            "exp_name":             exp_name,
        }
        last_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f"{exp_name}_last.pth")
        save_checkpoint(ckpt_state, last_ckpt_path)

        ckpt_tag = ""
        if avg_auc > best_val_auc:
            best_val_auc     = avg_auc
            best_epoch       = epoch
            patience_counter = 0
            best_ckpt_path   = os.path.join(config.CHECKPOINTS_DIR, f"{exp_name}_best.pth")
            save_checkpoint(ckpt_state, best_ckpt_path)
            ckpt_tag = " [best saved]"
        else:
            patience_counter += 1

        # ── Epoch summary line ────────────────────────────────────────────────
        tqdm.write(
            f"Ep {epoch:03d}/{config.NUM_EPOCHS} | "
            f"TrainL {train_loss:.4f} | ValL {val_loss:.4f} | "
            f"AUC {avg_auc:.4f} | Best {best_val_auc:.4f} @ep{best_epoch:03d}"
            + ckpt_tag
        )

        # Per-class AUC table every 5 epochs and on the final epoch
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            tqdm.write(f"\n  {'Class':<22} {'Val AUC':>8}")
            tqdm.write(f"  {'-'*32}")
            for cls, auc in per_class_auc.items():
                auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
                tqdm.write(f"  {cls:<22} {auc_str:>8}")
            tqdm.write("")

        epoch_bar.set_postfix(auc=f"{avg_auc:.4f}", best=f"{best_val_auc:.4f}", refresh=False)

        if patience_counter >= config.PATIENCE:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch}.")
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
