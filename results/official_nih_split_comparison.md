# Official NIH Split Comparison — DenseNet121 + CBAM block34 + BCE + AdamW

This document compares the best custom-split model (Experiment 08) against the same
architecture/training configuration retrained and evaluated on the **official NIH
ChestX-ray14 split** (`train_val_list.txt` / `test_list.txt`), and against a
referenced literature result (Hanif et al.).

- Experiment 08 (custom split) was trained on a patient-level 70/15/15 split
  (`splits/train_split.csv`, `splits/val_split.csv`, `splits/test_split.csv`).
- The official-split experiment (`densenet121_cbam_block34_bce_adamw_official_split`)
  uses NIH's official `test_list.txt` as the test set, with `train_val_list.txt`
  split patient-level 90/10 into train/val (`splits/official_nih/`). No patient
  appears in more than one of train/val/test.
- **Exp08's checkpoint was NOT evaluated on the official test split** — it was
  trained on the custom split and would leak information (its train set overlaps
  with the official test set), so a separate model was trained from scratch on the
  official split.

## Config (identical for both runs)

DenseNet121 + CBAM block34 + BCE + AdamW, no CLAHE
LR=1e-4, batch size=64, weight_decay=1e-5, max_epochs=50, patience=7, ReduceLROnPlateau.

## Results

| Split | Mean Test AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Exp08 — custom 70/15/15 split | 0.8371 | 0.4482 | 0.1395 | 0.1962 |
| Official NIH split (`..._official_split`) | TBD | TBD | TBD | TBD |
| Hanif et al. (reference, avg AUC) | 0.8096 | — | — | — |

*TBD values are filled in from `results/metrics/densenet121_cbam_block34_bce_adamw_official_split_test_summary.csv`
after `python train.py` and `python evaluate.py` have been run with `SPLIT_MODE = "official_nih"`.*

## Per-class AUC

| Class | Exp08 (custom split) | Official split | Hanif et al. |
|---|---|---|---|
| Atelectasis | 0.8169 | TBD | — |
| Cardiomegaly | 0.9026 | TBD | — |
| Effusion | 0.8877 | TBD | — |
| Infiltration | 0.7157 | TBD | — |
| Mass | 0.8452 | TBD | — |
| Nodule | 0.7494 | TBD | — |
| Pneumonia | 0.7595 | TBD | — |
| Pneumothorax | 0.8841 | TBD | — |
| Consolidation | 0.7998 | TBD | — |
| Edema | 0.9063 | TBD | — |
| Emphysema | 0.9214 | TBD | — |
| Fibrosis | 0.8102 | TBD | — |
| Pleural_Thickening | 0.7959 | TBD | — |
| Hernia | 0.9246 | TBD | — |
| **Mean** | **0.8371** | **TBD** | **0.8096** |

*Official-split per-class values come from
`results/metrics/densenet121_cbam_block34_bce_adamw_official_split_test_auc_per_class.csv`.*

## Interpretation

TBD — once the official-split run completes, summarize:
- Whether Mean Test AUC on the official split is higher, lower, or comparable to
  Exp08's custom-split result (0.8371), and whether the gap is consistent with the
  official test set's different patient/finding distribution.
- How the official-split result compares to Hanif et al.'s reference average AUC
  (0.8096) under the same (official) split protocol.
- Any notable per-class shifts, particularly for harder classes (Infiltration,
  Nodule, Pneumonia, Pleural_Thickening, Consolidation).
