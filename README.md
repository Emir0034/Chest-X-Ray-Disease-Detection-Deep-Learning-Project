# Chest X-Ray Disease Detection Using DenseNet121 with Attention and Loss Function Analysis

Multi-label classification of 14 chest diseases from frontal X-ray images using DenseNet121 on the NIH ChestX-ray14 dataset. Systematic ablation over CLAHE preprocessing, CBAM attention placement, loss functions (BCE, ASL, Focal Loss), and optimizers (Adam, AdamW).

---

## Project Overview

This project investigates how different design choices affect multi-label chest X-ray classification performance. It uses DenseNet121 as the backbone and evaluates four design axes through 14 controlled experiments:

- **CLAHE preprocessing:** on vs. off
- **CBAM attention placement:** block4 only / block34 / block1234
- **Loss function:** BCE / Asymmetric Loss (ASL) / Focal Loss (γ=2)
- **Optimizer:** Adam vs. AdamW

The primary metric is Mean Test AUC (macro-average ROC-AUC over all 14 disease classes), which is threshold-independent and robust to class imbalance. Precision, Recall, and F1 are reported at a fixed threshold of 0.5.

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal-view chest X-rays from 30,805 unique patients. Each image carries up to 14 disease labels extracted from radiology reports using NLP.

- **Labels:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
- **Task:** Multi-label classification — a single image can show multiple diseases simultaneously
- **Split:** Patient-level 70% train / 15% validation / 15% test (no patient appears in more than one split)
- **Label type:** Image-level only; no bounding box annotations

Download from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) and place images under `data/NIH Chest X-rays/`. The label CSV (`Data_Entry_2017.csv`) should also be placed there.

---

## Task Description: Multi-Label Classification

Each image receives 14 independent binary predictions, one per disease class. **Sigmoid is used instead of softmax** because a patient can have multiple diseases simultaneously. Softmax forces predictions to sum to 1 (single-label assumption), which is incorrect for this task.

---

## Methodology

### Backbone

ImageNet-pretrained DenseNet121 (CheXNet-style). The original classifier is replaced with a 1024→14 linear layer followed by sigmoid outputs (one per disease class). Raw logits are used during training; sigmoid probabilities are used at inference.

### CBAM Attention

Convolutional Block Attention Module (CBAM) applies sequential channel and spatial attention. Three insertion positions are tested:

- **block4 only** — after DenseBlock4, before Global Average Pooling (7×7 feature maps)
- **block34** — after DenseBlock3 and after DenseBlock4 (14×14 and 7×7 feature maps) ← best
- **block1234** — after all four DenseBlocks

### Loss Functions

- **BCE:** standard binary cross-entropy (baseline)
- **Asymmetric Loss (ASL):** separate focusing parameters for positive (γ+=0.0) and negative (γ−=2.0) samples, with a probability shift of 0.05 to suppress easy negatives
- **Focal Loss:** symmetric focusing with γ=2.0

### Optimizers and Training

- Adam and AdamW (decoupled weight decay), learning rate 1×10⁻⁴, weight decay 1×10⁻⁵
- ReduceLROnPlateau scheduler on validation AUC
- Early stopping: patience 7 epochs on validation AUC
- Batch size 64, up to 50 epochs, mixed-precision (AMP)
- Augmentation: RandomResizedCrop, RandomRotation ±10°, ColorJitter. No horizontal flips (anatomically inappropriate).

---

## Main Experiments

All values are Mean Test AUC and threshold-0.5 metrics from the held-out test split.

| Exp | Configuration | AUC | Precision | Recall | F1 |
|-----|--------------|-----|-----------|--------|-----|
| 01 | DenseNet121 + BCE | 0.8352 | 0.4226 | 0.1181 | 0.1700 |
| 02 | + CLAHE | 0.8339 | 0.4195 | 0.1310 | 0.1843 |
| 03a | + CBAM block4 + BCE | 0.8323 | 0.4465 | 0.1274 | 0.1864 |
| 03b | + CBAM block34 + BCE | 0.8359 | 0.4044 | 0.1050 | 0.1451 |
| 03c | + CBAM block1234 + BCE | 0.8343 | 0.3507 | 0.0912 | 0.1345 |
| 07 | CBAM block34 + BCE + Adam (no CLAHE) | 0.8361 | 0.4421 | 0.1284 | 0.1790 |
| **08** | **CBAM block34 + BCE + AdamW (no CLAHE)** | **0.8371** | **0.4482** | **0.1395** | **0.1962** |
| **09** | **CBAM block34 + ASL + AdamW (no CLAHE)** | **0.8370** | **0.3865** | **0.2385** | **0.2736** |
| 10 | CBAM block34 + Focal Loss + AdamW | 0.8351 | 0.4391 | 0.1258 | 0.1769 |

All models use DenseNet121 on NIH ChestX-ray14 with patient-level 70/15/15 split, threshold 0.5 for Precision/Recall/F1.

---

## Key Results

**Best AUC model — Experiment 08**
- Configuration: DenseNet121 + CBAM block34 + BCE + AdamW, no CLAHE
- Mean Test AUC: **0.8371**
- Precision: 0.4482 | Recall: 0.1395 | F1: 0.1962

**Best Recall/F1 model — Experiment 09**
- Configuration: DenseNet121 + CBAM block34 + ASL + AdamW, no CLAHE
- Mean Test AUC: **0.8370**
- Precision: 0.3865 | Recall: **0.2385** | F1: **0.2736**

Switching BCE → ASL in Experiment 09 changes AUC by only −0.0001 while improving Recall by **+0.0990** and F1 by **+0.0774**, at the cost of lower Precision (−0.0617). This reflects a Precision-Recall trade-off at threshold 0.5.

**Use Exp08** when Mean Test AUC is the priority. **Use Exp09** when sensitivity (detecting as many true positives as possible) matters more.

---

## Main Findings

- **CBAM placement** — block34 was the best: block34 (0.8359) > block1234 (0.8343) > block4 (0.8323)
- **CLAHE** — did not consistently improve Mean Test AUC; final best models do not use CLAHE
- **AdamW** — small but consistent improvement over Adam (+0.0010 AUC, Exp07→Exp08)
- **ASL** — substantially improved Recall (+0.0990) and F1 (+0.0774) at near-zero AUC cost
- **Focal Loss** — did not improve over BCE or ASL in any metric
- **FAAR** (Frequency-Aware Attention Refinement) — tested in Exp05/05b; did not improve AUC and was dropped
- **Reference comparison:** Exp08 achieved higher AUC than a referenced recent study in 12/14 disease classes, including 4/5 difficult classes. This is treated as a reference comparison only because evaluation protocols may differ.

---

## Grad-CAM Qualitative Analysis

Grad-CAM visualizations were generated for selected positive test examples to inspect model behavior qualitatively. For a Hernia example, Experiment 09 (ASL) produced a broader activation region with predicted probability 0.33, while Experiment 08 predicted 0.04 — consistent with the higher recall-oriented behavior of ASL.

Grad-CAM is used for qualitative inspection only. It is not localization proof. The NIH ChestX-ray14 dataset provides image-level labels only, with no bounding box annotations. Heatmap activations cannot be validated against ground-truth disease locations.

---

## Project Structure

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
├── requirements.txt
├── splits/                     # Auto-generated patient-level CSV splits
    ├── train_split.csv
    ├── val_split.csv
    └── test_split.csv

```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the experiment

Edit `config.py` to set:

- `DATA_CSV` — path to `Data_Entry_2017.csv`
- `IMAGES_DIR` — path to the image folder
- `USE_CBAM`, `CBAM_PLACEMENT`, `USE_ASYMMETRIC_LOSS`, `USE_FOCAL_LOSS`, `OPTIMIZER_NAME`
- `EXPERIMENT_STATUS = "official"` for official runs, `"trial"` for exploratory runs

### 3. Train

```bash
python train.py
```
Training saves checkpoints and metrics under a generated results/ folder.

### 4. Evaluate

```bash
python evaluate.py
```
Evaluation requires a trained checkpoint under results/checkpoints/. Metrics are saved under results/metrics/.

Loads the best checkpoint for the current `config.py` configuration and evaluates on the test split. Saves per-class AUC, summary CSV, and full metrics JSON.

---

## Future Work

- **Class-specific threshold tuning:** optimizing per-class decision thresholds on the validation set to improve the Precision-Recall balance for each disease
- **External validation:** evaluating on CheXpert or MIMIC-CXR to assess generalization
- **Alternative backbones:** testing EfficientNet, ResNet variants, or vision transformers to determine whether the observed findings generalize beyond DenseNet121

---

*This project was completed as an academic deep learning course project. It is not intended for clinical use.*
