# Chest X-Ray Disease Detection

Multi-label classification of 14 chest diseases from frontal X-ray images using DenseNet121 on the NIH ChestX-ray14 dataset. Systematic ablation over CLAHE preprocessing, CBAM attention, loss functions (BCE, ASL, Focal Loss), and optimizers (Adam, AdamW).

> **Disclaimer:** This is an academic deep learning research project. It is not intended for clinical use and should not be used for medical diagnosis.

---

## Overview

- **Task:** 14-class multi-label chest X-ray disease classification
- **Dataset:** NIH ChestX-ray14 (112,120 frontal X-ray images)
- **Backbone:** DenseNet121 (ImageNet pretrained)
- **Framework:** PyTorch with AMP mixed-precision training
- **Primary metric:** Mean Test AUC (macro ROC-AUC over 14 classes)
- **Secondary metrics:** Precision, Recall, F1 at threshold 0.5

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal-view chest X-rays with 14 disease labels extracted via NLP from radiology reports. Each image may have zero or more labels (multi-label format).

Download the dataset from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) and place images under `data/NIH Chest X-rays/`. The label CSV (`Data_Entry_2017.csv`) should also be placed there.

Patient-level splits (70% train / 15% val / 15% test) are generated automatically and saved to `splits/` on first run.

---

## Disease Classes

| | | |
|---|---|---|
| Atelectasis | Cardiomegaly | Effusion |
| Infiltration | Mass | Nodule |
| Pneumonia | Pneumothorax | Consolidation |
| Edema | Emphysema | Fibrosis |
| Pleural\_Thickening | Hernia | |

---

## Methods

| Component | Variants Tested |
|---|---|
| Preprocessing | Standard resize + augmentation; CLAHE contrast enhancement |
| Attention | No CBAM; CBAM after block4; block34 (best); block1234 |
| Loss function | BCEWithLogitsLoss; Asymmetric Loss (ASL); Focal Loss γ=2 |
| Optimizer | Adam; AdamW |
| Backbone | DenseNet121 |

**CBAM (Convolutional Block Attention Module):** Sequential channel and spatial attention gates. Tested at three insertion points inside DenseNet121.

**Asymmetric Loss (ASL):** Separate focusing parameters for positive and negative samples (γ\_neg=2, γ\_pos=0, shift=0.05). Addresses class imbalance by suppressing easy negatives more aggressively than Focal Loss.

**Augmentation:** RandomResizedCrop, RandomRotation ±10°, ColorJitter. No horizontal flips (anatomically inappropriate for chest X-rays).

---

## Results

### Best Models

| Model | Config | Mean Test AUC | Precision | Recall | F1 |
|---|---|:---:|:---:|:---:|:---:|
| Exp01 — Baseline | DenseNet121 + BCE | 0.8352 | 0.4226 | 0.1181 | 0.1700 |
| **Exp08 — Best AUC** | DenseNet121 + CBAM block34 + BCE + AdamW | **0.8371** | 0.4482 | 0.1395 | 0.1962 |
| **Exp09 — Best Recall/F1** | DenseNet121 + CBAM block34 + ASL + AdamW | 0.8370 | 0.3865 | **0.2385** | **0.2736** |
| Exp10 — Focal Loss | DenseNet121 + CBAM block34 + Focal γ=2 + AdamW | 0.8351 | 0.4391 | 0.1258 | 0.1769 |

All models: no CLAHE, patient-level 70/15/15 split, threshold 0.5 for Precision/Recall/F1.

### Trial Result

| Model | Config | Mean Test AUC | Precision | Recall | F1 |
|---|---|:---:|:---:|:---:|:---:|
| Stage 2 ASL fine-tune (trial) | Exp08 weights → ASL re-train, lr=2e-5 | 0.8320 | 0.3528 | 0.2944 | 0.3083 |

Trial result only — improved Recall/F1 but reduced AUC and Precision relative to Exp09. Not promoted as an official final model.

### Per-Class AUC — Best Models (Exp08 vs Exp09)

Per-class test AUC figures are saved in `results/figures/final_comparisons/`.

---

## Main Findings

- **CLAHE** did not improve overall Mean Test AUC; per-class effects were mixed. Removed from the final architecture.
- **CBAM placement:** block34 (after DenseBlock3 and DenseBlock4) outperformed block4-only and block1234. Deeper attention placement with limited scope worked best.
- **Removing CLAHE** while keeping CBAM block34 improved results. The CLAHE+CBAM combination was not beneficial.
- **AdamW** provided a small but consistent improvement over Adam (+0.0010 Mean Test AUC) via decoupled weight decay.
- **ASL (Exp09):** Nearly identical AUC to BCE (0.8370 vs 0.8371), but substantially higher Recall (+0.0990) and F1 (+0.0774). Recommended when sensitivity is more important than ranking.
- **Focal Loss γ=2 (Exp10):** Did not improve Mean Test AUC, Recall, or F1 over BCE or ASL.
- **FAAR** (Frequency-Aware Attention Refinement) did not improve Mean Test AUC and was discontinued after two tuning variants.

**Recommendation:**
- Use **Exp08** when Mean Test AUC is the primary goal.
- Use **Exp09** when Recall/F1 (sensitivity) is more important.

---

## Repository Structure

```
├── config.py                   # All hyperparameters, flags, experiment naming
├── train.py                    # Training loop with AMP, early stopping, checkpointing
├── evaluate.py                 # Test-set evaluation and metric export
├── model.py                    # DenseNet121 + optional CBAM + optional FAAR
├── dataset.py                  # NIH dataset loading, patient-level splits
├── loss.py                     # AsymmetricLoss, FocalLoss, BCE wrapper
├── transforms.py               # CLAHE, train/val augmentation pipelines
├── cbam.py                     # CBAM channel + spatial attention modules
├── faar.py                     # FAAR frequency-aware attention module
├── utils.py                    # Seeding, checkpoint I/O, AUC computation, plotting
├── gradcam.py                  # Grad-CAM hook implementation
├── gradcam_compare.py          # Side-by-side Grad-CAM for Exp08 vs Exp09
├── plot_final_comparisons.py   # Generate all comparison figures (14 experiments)
├── EXPERIMENTS.md              # Detailed experiment log with all results
├── splits/                     # Auto-generated patient-level CSV splits
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── results/
│   ├── checkpoints/            # Official model checkpoints (not tracked in git)
│   ├── metrics/                # CSV and JSON metrics for all experiments
│   ├── figures/                # Training curves, AUC bar charts, comparisons
│   │   ├── final_comparisons/  # Cross-experiment comparison figures
│   │   └── gradcam/            # Grad-CAM overlay images
│   └── trials/                 # Trial run outputs (isolated from official results)
└── report/                     # LaTeX academic report
    ├── main.tex
    ├── references.bib
    ├── sections/
    └── figures/
```

---

## Setup

Install the required packages:

```bash
pip install torch torchvision
pip install pandas scikit-learn matplotlib opencv-python Pillow numpy tqdm
```

Then set the dataset paths in `config.py`:

```python
DATA_CSV   = r"data/NIH Chest X-rays/Data_Entry_2017.csv"
IMAGES_DIR = r"data/NIH Chest X-rays"
```

---

## Training

Configure the experiment in `config.py` (set flags for `USE_CBAM`, `USE_ASYMMETRIC_LOSS`, `OPTIMIZER_NAME`, etc.), then run:

```bash
python train.py
```

**Experiment mode:** Set `EXPERIMENT_STATUS = "trial"` in `config.py` for exploratory runs. Trial outputs are saved to `results/trials/` and the experiment name gets a `_trial` suffix.

**Fine-tuning:** Set `FINE_TUNE_FROM_CHECKPOINT` in `config.py` to initialize model weights from an existing checkpoint. Optimizer, scheduler, epoch counter, and best AUC are always freshly initialized.

Checkpoints are saved to `results/checkpoints/` (official) or `results/trials/checkpoints/` (trial).

---

## Evaluation

```bash
python evaluate.py
```

Loads the best checkpoint for the current `config.py` configuration and evaluates on the test set. Saves:
- Per-class AUC CSV and bar chart
- Test summary CSV
- Full metrics JSON

---

## Grad-CAM Visualization

`gradcam_compare.py` generates side-by-side Grad-CAM heatmaps comparing:
- **Exp08** — best Mean Test AUC model (BCE + AdamW)
- **Exp09** — best Recall/F1 model (ASL + AdamW)

Find positive test examples for a class:

```bash
python gradcam_compare.py --find_examples --class_name Hernia --max_examples 5
```

Generate a Grad-CAM comparison for a specific image:

```bash
python gradcam_compare.py --image "path/to/image.png" --class_name Hernia --output_name hernia_example_01
```

Outputs (original, Exp08 overlay, Exp09 overlay, and side-by-side comparison) are saved to `results/figures/gradcam/`.

> Grad-CAM is used for qualitative interpretability only. Heatmap activations should not be interpreted as definitive localization or diagnostic evidence.

---

## Experiment Tracking

All experiments are logged in `EXPERIMENTS.md` with full configuration, training curves, per-class AUC tables, and observations. Comparison figures across all 14 experiments are generated by `plot_final_comparisons.py`.

---

## Excluded Files

The following are not tracked in this repository:

| Item | Reason |
|---|---|
| `data/` | NIH ChestX-ray14 (~45 GB) — download separately |
| `results/checkpoints/` | Large model checkpoint files |
| `results/trials/` | Trial run outputs |
| Compiled Python cache | Auto-generated |

---

## References

- Rajpurkar et al. (2017). CheXNet. arXiv:1711.05225
- Wang et al. (2017). NIH ChestX-ray14. CVPR.
- Woo et al. (2018). CBAM. ECCV.
- Ridnik et al. (2021). Asymmetric Loss. ICCV.
- Al-Saggaf et al. (2025). Multi-Label CXR Classification. Bioengineering.

---

*This project was completed as an academic deep learning course project. It is not intended for clinical use.*
