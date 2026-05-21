# EXPERIMENTS — Chest X-Ray Disease Detection Project

Bu dosya her deneyin sonuçlarını, gözlemlerini ve karşılaştırmalarını tutar.
- **Val AUC** → train.py çıktısı (eğitim sırasında)
- **Test AUC** → evaluate.py çıktısı (asıl final sonuç)
- Kod değişiklikleri için CHANGES.md'ye bakın.

---

## [Deney 01] — Baseline
- **Experiment Name:** `densenet121_bce`
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = False
  - USE_CBAM = False
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8319
  - Best Epoch: 6
  - Train Loss (epoch 6): 0.1351
  - Val Loss (epoch 6): 0.1470
  - Early stopping triggered at epoch 13
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: results/checkpoints/densenet121_bce_best.pth
  - Mean Test AUC: 0.8352
  - Precision: 0.4226
  - Recall: 0.1181
  - F1 Score: 0.1700
  - Threshold: 0.5
  - Per-class Test AUC:
    | Sınıf | Test AUC |
    |---|---|
    | Atelectasis | 0.8115 |
    | Cardiomegaly | 0.8990 |
    | Effusion | 0.8836 |
    | Infiltration | 0.7107 |
    | Mass | 0.8336 |
    | Nodule | 0.7696 |
    | Pneumonia | 0.7548 |
    | Pneumothorax | 0.8761 |
    | Consolidation | 0.8000 |
    | Edema | 0.9045 |
    | Emphysema | 0.9192 |
    | Fibrosis | 0.8210 |
    | Pleural_Thickening | 0.7845 |
    | Hernia | 0.9247 |
- **Dosyalar:**
  - Loss CSV: results/metrics/densenet121_bce_loss_history.csv
  - Val AUC CSV: results/metrics/densenet121_bce_val_auc_history.csv
  - Best Epoch JSON: results/metrics/densenet121_bce_best_epoch.json
  - Test AUC CSV: results/metrics/densenet121_bce_test_auc_per_class.csv
  - Test Summary CSV: results/metrics/densenet121_bce_test_summary.csv
  - Test Metrics JSON: results/metrics/densenet121_bce_test_metrics.json
  - Curves PNG: results/figures/densenet121_bce_curves.png
  - Test AUC Bar PNG: results/figures/densenet121_bce_test_auc_bar.png
- **Gözlemler:**
  - Best Val AUC = 0.8319 at epoch 6; early stopping at epoch 13.
  - Training loss continued to fall after epoch 6 while validation AUC declined — overfitting.
  - Test AUC = 0.8352 ≈ Val AUC → baseline generalisation is acceptable.
  - Precision/Recall/F1 are low at threshold 0.5; AUC is the primary comparison metric.
  - Strongest: Hernia (0.9247), Emphysema (0.9192), Edema (0.9045), Cardiomegaly (0.8990).
  - Weakest: Infiltration (0.7107), Pneumonia (0.7548), Nodule (0.7696), Pleural_Thickening (0.7845).
- **Sonraki adım:** Run Experiment 02 — DenseNet121 + CLAHE + BCE.

---

## [Deney 02] — +CLAHE
- **Experiment Name:** `densenet121_clahe_bce`
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = True
  - USE_CBAM = False
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8321
  - Best Epoch: 6
  - Train Loss (epoch 6): 0.1342
  - Val Loss (epoch 6): 0.1475
  - Early stopping triggered at epoch 13
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_bce_best.pth`
  - Mean Test AUC: 0.8339
  - Deney 01'e göre fark: −0.0013
  - Precision: 0.4195
  - Recall: 0.1310
  - F1 Score: 0.1843
  - Threshold: 0.5
  - Per-class Test AUC:
    | Sınıf | Test AUC | Deney 01 Farkı |
    |---|---|---|
    | Atelectasis | 0.8115 | 0.0000 |
    | Cardiomegaly | 0.8999 | +0.0009 |
    | Effusion | 0.8839 | +0.0003 |
    | Infiltration | 0.7103 | −0.0004 |
    | Mass | 0.8388 | +0.0052 |
    | Nodule | 0.7602 | −0.0094 |
    | Pneumonia | 0.7456 | −0.0092 |
    | Pneumothorax | 0.8809 | +0.0048 |
    | Consolidation | 0.8039 | +0.0039 |
    | Edema | 0.9110 | +0.0065 |
    | Emphysema | 0.9181 | −0.0011 |
    | Fibrosis | 0.8101 | −0.0109 |
    | Pleural_Thickening | 0.7937 | +0.0092 |
    | Hernia | 0.9060 | −0.0187 |
- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_bce_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_bce_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_bce_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_bce_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_bce_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_bce_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_bce_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_bce_test_auc_bar.png`
- **Gözlemler:**
  - CLAHE preprocessing alone did not improve Mean Test AUC vs Baseline (0.8339 vs 0.8352, −0.0013).
  - Best epoch remained at 6, suggesting a similar epoch-wise convergence pattern; however, CLAHE increased preprocessing cost and training time significantly (~223s/epoch → ~371s/epoch).
  - Some classes improved (Edema +0.0065, Pleural_Thickening +0.0092), others declined (Hernia −0.0187, Fibrosis −0.0109).
  - Inconsistent per-class effect suggests preprocessing alone is insufficient.
  - CLAHE improved local contrast but the marginal and inconsistent AUC changes show it was insufficient as a standalone contribution. This motivates testing CBAM in Experiment 03 to evaluate whether attention can better exploit contrast-enhanced features.
- **Sonraki adım:** Run Experiment 03 — DenseNet121 + CLAHE + CBAM + BCE.

---

## [Deney 03a] — +CLAHE +CBAM (block4)

- **Experiment Name:** `densenet121_clahe_cbam_block4_bce`
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block4"
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **CBAM Placement:**
  - CBAM after DenseBlock4 / norm5 / ReLU, before Global Average Pooling.
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8301
  - Best Epoch: 6
  - Train Loss (epoch 6): 0.1347
  - Val Loss (epoch 6): 0.1485
  - Early stopping triggered at epoch 13
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block4_bce_best.pth`
  - Mean Test AUC: 0.8323
  - Deney 02'ye göre fark: −0.0016
  - Precision: 0.4465
  - Recall: 0.1274
  - F1 Score: 0.1864
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | Deney 02 Farkı |
    |---|---|---|
    | Atelectasis | 0.8130 | +0.0015 |
    | Cardiomegaly | 0.8936 | −0.0063 |
    | Effusion | 0.8862 | +0.0023 |
    | Infiltration | 0.7152 | +0.0049 |
    | Mass | 0.8378 | −0.0010 |
    | Nodule | 0.7596 | −0.0006 |
    | Pneumonia | 0.7564 | +0.0108 |
    | Pneumothorax | 0.8823 | +0.0014 |
    | Consolidation | 0.7996 | −0.0043 |
    | Edema | 0.8981 | −0.0129 |
    | Emphysema | 0.9197 | +0.0016 |
    | Fibrosis | 0.8058 | −0.0043 |
    | Pleural_Thickening | 0.7804 | −0.0133 |
    | Hernia | 0.9050 | −0.0010 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block4_bce_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block4_bce_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block4_bce_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block4_bce_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block4_bce_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block4_bce_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block4_bce_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block4_bce_test_auc_bar.png`
- **Gözlemler:**
  - CBAM at block4 did not improve Mean Test AUC vs Deney 02 (0.8323 vs 0.8339, −0.0016).
  - Best epoch remained at 6 — adding CBAM at this placement did not affect convergence speed.
  - Notable improvements: Pneumonia +0.0108, Infiltration +0.0049, Effusion +0.0023, Emphysema +0.0016.
  - Notable regressions: Pleural_Thickening −0.0133, Edema −0.0129, Cardiomegaly −0.0063.
  - The block4-only placement appears to offer mixed per-class effects with a slight net negative on mean AUC. Experiments 03b and 03c with multi-block CBAM placement may yield better results.
- **Sonraki adım:** Run Experiment 03b — CBAM at block34.

---

## [Deney 03b] — +CLAHE +CBAM (block34)

- **Experiment Name:** `densenet121_clahe_cbam_block34_bce`
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **CBAM Placement:**
  - CBAM after DenseBlock3 before Transition3.
  - CBAM after DenseBlock4 / norm5 / ReLU, before Global Average Pooling.
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8320
  - Best Epoch: 5
  - Train Loss (epoch 5): 0.1377
  - Val Loss (epoch 5): 0.1472
  - Early stopping triggered at epoch 12
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block34_bce_best.pth`
  - Mean Test AUC: 0.8359
  - Deney 02'ye göre fark: +0.0020
  - Precision: 0.4044
  - Recall: 0.1050
  - F1 Score: 0.1451
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | Deney 02 Farkı | 03a Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8088 | −0.0027 | −0.0042 |
    | Cardiomegaly | 0.8948 | −0.0051 | +0.0012 |
    | Effusion | 0.8858 | +0.0019 | −0.0004 |
    | Infiltration | 0.7129 | +0.0026 | −0.0023 |
    | Mass | 0.8377 | −0.0011 | −0.0001 |
    | Nodule | 0.7594 | −0.0008 | −0.0002 |
    | Pneumonia | 0.7624 | +0.0168 | +0.0060 |
    | Pneumothorax | 0.8856 | +0.0047 | +0.0033 |
    | Consolidation | 0.7967 | −0.0072 | −0.0029 |
    | Edema | 0.9006 | −0.0104 | +0.0025 |
    | Emphysema | 0.9275 | +0.0094 | +0.0078 |
    | Fibrosis | 0.8192 | +0.0091 | +0.0134 |
    | Pleural_Thickening | 0.7906 | −0.0031 | +0.0102 |
    | Hernia | 0.9201 | +0.0141 | +0.0151 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block34_bce_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_bce_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block34_bce_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_bce_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block34_bce_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block34_bce_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block34_bce_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block34_bce_test_auc_bar.png`
- **Gözlemler:**
  - block34 placement improved Mean Test AUC vs Deney 02 (0.8359 vs 0.8339, +0.0020) and vs 03a (0.8359 vs 0.8323, +0.0036).
  - Best Val AUC nearly matched Deney 02 (0.8320 vs 0.8321, −0.0001) and exceeded 03a (0.8301, +0.0019).
  - Best epoch shifted to 5 (from 6 in 03a), early stopping at epoch 12 — slightly faster convergence.
  - Strong improvements vs 02: Pneumonia +0.0168, Hernia +0.0141, Emphysema +0.0094, Fibrosis +0.0091.
  - Notable regressions vs 02: Edema −0.0104, Consolidation −0.0072, Atelectasis −0.0027.
  - block34 outperforms block4 on Mean Test AUC and improves several difficult classes, suggesting that adding attention to mid-to-high-level features is more useful than block4-only attention.
  - However, threshold-based metrics (Precision, Recall, F1) decreased compared to 03a and Deney 02 at threshold 0.5.
  - Final CBAM placement selection deferred until 03c results are available for full three-way comparison.
- **Sonraki adım:** Run Experiment 03c — CBAM_PLACEMENT = "block1234".

---

## [Deney 03c] — +CLAHE +CBAM (block1234)

- **Experiment Name:** `densenet121_clahe_cbam_block1234_bce`
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block1234"
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **CBAM Placement:**
  - CBAM after DenseBlock1 before Transition1.
  - CBAM after DenseBlock2 before Transition2.
  - CBAM after DenseBlock3 before Transition3.
  - CBAM after DenseBlock4 / norm5 / ReLU, before Global Average Pooling.
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8311
  - Best Epoch: 5
  - Train Loss (epoch 5): 0.1384
  - Val Loss (epoch 5): 0.1470
  - Early stopping triggered at epoch 12
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block1234_bce_best.pth`
  - Mean Test AUC: 0.8343
  - Deney 02'ye göre fark: +0.0004
  - 03b'ye göre fark: −0.0016
  - Precision: 0.3507
  - Recall: 0.0912
  - F1 Score: 0.1345
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | Deney 02 Farkı | 03b Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8173 | +0.0058 | +0.0085 |
    | Cardiomegaly | 0.9035 | +0.0036 | +0.0087 |
    | Effusion | 0.8806 | −0.0033 | −0.0052 |
    | Infiltration | 0.7121 | +0.0018 | −0.0008 |
    | Mass | 0.8361 | −0.0027 | −0.0016 |
    | Nodule | 0.7638 | +0.0036 | +0.0044 |
    | Pneumonia | 0.7564 | +0.0108 | −0.0060 |
    | Pneumothorax | 0.8827 | +0.0018 | −0.0029 |
    | Consolidation | 0.8017 | −0.0022 | +0.0050 |
    | Edema | 0.8996 | −0.0114 | −0.0010 |
    | Emphysema | 0.9211 | +0.0030 | −0.0064 |
    | Fibrosis | 0.8061 | −0.0040 | −0.0131 |
    | Pleural_Thickening | 0.7830 | −0.0107 | −0.0076 |
    | Hernia | 0.9165 | +0.0105 | −0.0036 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block1234_bce_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block1234_bce_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block1234_bce_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block1234_bce_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block1234_bce_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block1234_bce_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block1234_bce_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block1234_bce_test_auc_bar.png`
- **Gözlemler:**
  - block1234 slightly improved Mean Test AUC vs Deney 02 (0.8343 vs 0.8339, +0.0004), but the improvement was very small.
  - Best Val AUC was lower than Deney 02 (0.8311 vs 0.8321, −0.0010).
  - block1234 did not outperform block34: Val AUC decreased from 0.8320 to 0.8311 (−0.0009) and Mean Test AUC decreased from 0.8359 to 0.8343 (−0.0016).
  - Best epoch remained 5 and early stopping occurred at epoch 12 — same convergence pattern as 03b.
  - Strong improvements vs Deney 02: Pneumonia +0.0108, Hernia +0.0105, Atelectasis +0.0058, Cardiomegaly +0.0036, Nodule +0.0036.
  - Notable regressions vs Deney 02: Edema −0.0114, Pleural_Thickening −0.0107, Fibrosis −0.0040, Effusion −0.0033.
  - Compared to 03b, block1234 improved Atelectasis (+0.0085) and Cardiomegaly (+0.0087) but decreased Fibrosis (−0.0131), Pleural_Thickening (−0.0076), Emphysema (−0.0064), Pneumonia (−0.0060), and Hernia (−0.0036).
  - Threshold-based metrics were the weakest among CBAM placement experiments (F1: 0.1345).
  - Adding CBAM to all DenseBlocks did not provide a consistent benefit and may have over-attenuated early feature representations.
  - Based on the full CBAM placement ablation, block34 is the strongest candidate for Experiment 04 and Experiment 05.
- **Sonraki adım:** Select CBAM_PLACEMENT = "block34" for Experiment 04 — CLAHE + CBAM block34 + ASL.

---

### CBAM Placement Decision

- **Selected CBAM Placement:** block34
- **Reason:**
  - 03a block4: Best Val AUC 0.8301, Mean Test AUC 0.8323
  - 03b block34: Best Val AUC 0.8320, Mean Test AUC 0.8359 ← best on both metrics
  - 03c block1234: Best Val AUC 0.8311, Mean Test AUC 0.8343
  - block34 achieved the highest validation AUC and the highest Mean Test AUC among all three CBAM placement experiments.
  - block1234 added parameters without consistent improvement, suggesting over-attenuation of early feature representations.
- **Decision Note:**
  - CBAM_PLACEMENT = "block34" is fixed for Experiment 04 and Experiment 05 so that later improvements can be attributed mainly to ASL and FAAR instead of changing the attention location.

---

## [Deney 04a] — +CLAHE +CBAM +ASL
- **Experiment Name:** `densenet121_clahe_cbam_block34_asl`
- **ASL Setting:** gn4_gp1
- **Config:**
  - Model: DenseNet121
  - Loss: AsymmetricLoss
  - ASL parameters: gamma_neg=4.0, gamma_pos=1.0, clip=0.05
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = True
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8273
  - Best Epoch: 6
  - Train Loss (epoch 6): 0.3282
  - Val Loss (epoch 6): 0.3623
  - Early stopping triggered at epoch 13
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block34_asl_best.pth`
  - Mean Test AUC: 0.8305
  - Deney 03b'ye göre fark: −0.0054
  - Precision: 0.2686
  - Recall: 0.4392
  - F1 Score: 0.3250
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | 03b Farkı |
    |---|---|---|
    | Atelectasis | 0.8150 | +0.0062 |
    | Cardiomegaly | 0.8982 | +0.0034 |
    | Effusion | 0.8842 | −0.0016 |
    | Infiltration | 0.7115 | −0.0014 |
    | Mass | 0.8403 | +0.0026 |
    | Nodule | 0.7478 | −0.0116 |
    | Pneumonia | 0.7348 | −0.0276 |
    | Pneumothorax | 0.8751 | −0.0105 |
    | Consolidation | 0.7873 | −0.0094 |
    | Edema | 0.8953 | −0.0053 |
    | Emphysema | 0.9288 | +0.0013 |
    | Fibrosis | 0.7995 | −0.0197 |
    | Pleural_Thickening | 0.7828 | −0.0078 |
    | Hernia | 0.9271 | +0.0070 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block34_asl_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block34_asl_test_auc_bar.png`
- **Gözlemler:**
  - ASL with gn4_gp1 decreased Mean Test AUC compared with Experiment 03b by −0.0054 (0.8305 vs 0.8359).
  - ASL with gn4_gp1 decreased Best Val AUC compared with Experiment 03b by −0.0047 (0.8273 vs 0.8320).
  - The largest per-class AUC regressions compared with Experiment 03b were Pneumonia (−0.0276), Fibrosis (−0.0197), and Nodule (−0.0116).
  - Gains compared with Experiment 03b were limited to a few classes: Atelectasis (+0.0062) and Hernia (+0.0070). Hernia was already one of the high-AUC classes.
  - Target rare classes such as Pneumonia, Nodule, and Fibrosis did not benefit from this ASL configuration.
  - ASL substantially increased Recall from 0.1050 to 0.4392 at threshold 0.5.
  - ASL increased F1 Score from 0.1451 to 0.3250 at threshold 0.5.
  - The model became more recall-oriented: it detected more positive cases but produced more false positives, lowering Precision from 0.4044 to 0.2686.
  - gamma_neg=4.0 may be too aggressive for this dataset because it may suppress easy negative examples too strongly and reduce useful signal for rare classes.
- **Sonraki adım:** Run Experiment 04b — ASL with a less aggressive setting: gamma_neg=2.0, gamma_pos=0.0.

---

## [Deney 04b] — +CLAHE +CBAM +ASL (gn2_gp0)
- **Experiment Name:** `densenet121_clahe_cbam_block34_asl_gn2_gp0`
- **ASL Setting:** gn2_gp0
- **Config:**
  - Model: DenseNet121
  - Loss: AsymmetricLoss
  - ASL parameters: gamma_neg=2.0, gamma_pos=0.0, clip=0.05
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = True
  - USE_FAAR = False
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8275
  - Best Epoch: 5
  - Train Loss (epoch 5): 0.8212
  - Val Loss (epoch 5): 0.8718
  - Early stopping triggered at epoch 12
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block34_asl_gn2_gp0_best.pth`
  - Mean Test AUC: 0.8342
  - Deney 04a'ya göre fark: +0.0037
  - Deney 03b'ye göre fark: −0.0017
  - Precision: 0.3740
  - Recall: 0.2313
  - F1 Score: 0.2604
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | 03b Farkı | 04a Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8086 | −0.0002 | −0.0064 |
    | Cardiomegaly | 0.8975 | +0.0027 | −0.0007 |
    | Effusion | 0.8822 | −0.0036 | −0.0020 |
    | Infiltration | 0.7166 | +0.0037 | +0.0051 |
    | Mass | 0.8350 | −0.0027 | −0.0053 |
    | Nodule | 0.7520 | −0.0074 | +0.0042 |
    | Pneumonia | 0.7557 | −0.0067 | +0.0209 |
    | Pneumothorax | 0.8834 | −0.0022 | +0.0083 |
    | Consolidation | 0.7994 | +0.0027 | +0.0121 |
    | Edema | 0.9000 | −0.0006 | +0.0047 |
    | Emphysema | 0.9277 | +0.0002 | −0.0011 |
    | Fibrosis | 0.8091 | −0.0101 | +0.0096 |
    | Pleural_Thickening | 0.7815 | −0.0091 | −0.0013 |
    | Hernia | 0.9308 | +0.0107 | +0.0037 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block34_asl_gn2_gp0_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block34_asl_gn2_gp0_test_auc_bar.png`
- **Gözlemler:**
  - ASL gn2_gp0 recovered most of the AUC loss observed in Experiment 04a: Mean Test AUC increased from 0.8305 to 0.8342 (+0.0037).
  - Compared with Experiment 03b, 04b still has slightly lower Mean Test AUC (0.8342 vs 0.8359, −0.0017) and Best Val AUC (0.8275 vs 0.8320, −0.0045).
  - However, 04b improved Recall (0.2313 vs 0.1050, +0.1263) and F1 (0.2604 vs 0.1451, +0.1153) compared with 03b while keeping Precision closer to 03b (0.3740 vs 0.4044).
  - Compared with 04a, 04b is more balanced: Precision improved (+0.1054) and the extreme Recall of 04a (0.4392) dropped to a more stable 0.2313. F1 score decreased from 0.3250 to 0.2604.
  - The strongest per-class gains vs 04a were Pneumonia (+0.0209), Consolidation (+0.0121), Pneumothorax (+0.0083), Fibrosis (+0.0096), and Infiltration (+0.0051).
  - Notable per-class regressions vs 03b: Fibrosis (−0.0101), Pleural_Thickening (−0.0091), Nodule (−0.0074), Pneumonia (−0.0067).
  - Hernia improved noticeably vs 03b (+0.0107).
  - Our ablation results across 04a and 04b show that gamma_neg=2.0, gamma_pos=0.0 is a better ASL setting than gamma_neg=4.0, gamma_pos=1.0 for this dataset: it yields higher AUC, better precision, and more stable per-class behaviour.
  - Our experiment results suggest that reducing gamma_neg from 4.0 to 2.0 and removing the positive focal term may reduce overly aggressive negative suppression and produce more stable ASL behaviour.
- **Sonraki adım:** Use ASL gn2_gp0 as the selected ASL setting for Experiment 05 together with FAAR.

---

## [Deney 05] — +CLAHE +CBAM +ASL +FAAR (Final)
- **Experiment Name:** `densenet121_clahe_cbam_block34_asl_gn2_gp0_faar`
- **Config:**
  - Model: DenseNet121
  - Loss: AsymmetricLoss
  - ASL Setting: gn2_gp0
  - ASL parameters: gamma_neg=2.0, gamma_pos=0.0, clip=0.05
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = True
  - USE_FAAR = True
  - FAAR_ALPHA_INIT = 0.0
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8288
  - Best Epoch: 5
  - Train Loss (epoch 5): 0.8225
  - Val Loss (epoch 5): 0.8787
  - Early stopping triggered at epoch 12
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_best.pth`
  - Mean Test AUC: 0.8312
  - Deney 04b'ye göre fark: −0.0030
  - Deney 03b'ye göre fark: −0.0047
  - Precision: 0.3793
  - Recall: 0.2567
  - F1 Score: 0.2788
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | 04b Farkı | Baseline Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8086 | +0.0000 | −0.0029 |
    | Cardiomegaly | 0.9001 | +0.0026 | +0.0011 |
    | Effusion | 0.8851 | +0.0029 | +0.0015 |
    | Infiltration | 0.7093 | −0.0073 | −0.0014 |
    | Mass | 0.8241 | −0.0109 | −0.0095 |
    | Nodule | 0.7671 | +0.0151 | −0.0025 |
    | Pneumonia | 0.7494 | −0.0063 | −0.0054 |
    | Pneumothorax | 0.8688 | −0.0146 | −0.0073 |
    | Consolidation | 0.8005 | +0.0011 | +0.0005 |
    | Edema | 0.8959 | −0.0041 | −0.0086 |
    | Emphysema | 0.9199 | −0.0078 | +0.0007 |
    | Fibrosis | 0.8040 | −0.0051 | −0.0170 |
    | Pleural_Thickening | 0.7759 | −0.0056 | −0.0086 |
    | Hernia | 0.9283 | −0.0025 | +0.0036 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_test_auc_bar.png`
- **Gözlemler:**
  - FAAR was designed to address class imbalance at the feature-attention level using class-frequency information, inserted after CBAM and before global average pooling.
  - FAAR improved Best Val AUC compared with Experiment 04b: 0.8288 vs 0.8275 (+0.0013).
  - However, FAAR decreased Mean Test AUC compared with Experiment 04b: 0.8312 vs 0.8342 (−0.0030).
  - FAAR improved Precision (+0.0053), Recall (+0.0254), and F1 (+0.0184) compared with Experiment 04b.
  - The strongest per-class improvement compared with 04b was Nodule (+0.0151).
  - Notable per-class regressions compared with 04b: Pneumothorax (−0.0146), Mass (−0.0109), Infiltration (−0.0073), Emphysema (−0.0078), Pleural_Thickening (−0.0056).
  - Compared with the Baseline (Experiment 01), Experiment 05 shows a net Mean Test AUC decrease (0.8312 vs 0.8352, −0.0040), but Recall and F1 are substantially higher.
  - Our experiment results indicate that FAAR partially helped sensitivity-related metrics but did not improve overall Mean Test AUC compared with 04b.
  - Based on observed validation and test AUC, FAAR can be discussed as a feature-level imbalance refinement that improves Recall and F1 while not improving Mean Test AUC relative to the best BCE configuration or the best ASL configuration.
- **Genel Değerlendirme:**
  - Best Mean Test AUC model: Experiment 08 (0.8371) — CBAM block34 + BCE + AdamW (no CLAHE). Experiment 03b remains the best CLAHE+CBAM BCE model (0.8359).
  - Best ASL-based model: Experiment 04b (0.8342) — CLAHE + CBAM block34 + ASL gn2_gp0.
  - Experiment 05 with FAAR achieved improved Best Val AUC (0.8288) and improved Recall/F1 compared with 04b, but did not surpass 04b on Mean Test AUC.
  - Our ablation results show that CBAM block34 placement is the single most impactful improvement over the baseline. ASL gn2_gp0 adds Recall and F1 benefits. FAAR further improves Recall but does not consistently improve AUC.
- **Sonraki adım:** Run Experiment 05b — FAAR cap3 tuning.

---

## [Deney 05b] — +CLAHE +CBAM +ASL +FAAR cap3

- **Experiment Name:** `densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3`
- **Config:**
  - Model: DenseNet121
  - Loss: AsymmetricLoss
  - ASL Setting: gn2_gp0
  - ASL parameters: gamma_neg=2.0, gamma_pos=0.0, clip=0.05
  - USE_CLAHE = True
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = True
  - USE_FAAR = True
  - FAAR_WEIGHT_MODE = "cap"
  - FAAR_CAP_MAX = 3.0
  - FAAR_ALPHA_INIT = 0.0
  - BATCH_SIZE = 64 | NUM_EPOCHS = 50 | LR = 1e-4 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **FAAR cap3 formula:**
  - freq = positive_count / total_samples
  - inv_freq = 1.0 / (freq + 1e-6)
  - freq_weights = inv_freq / inv_freq.mean()
  - freq_weights = np.minimum(freq_weights, 3.0)
  - freq_weights = freq_weights / freq_weights.mean()
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8282
  - Best Epoch: 5
  - Train Loss (epoch 5): 0.8222
  - Val Loss (epoch 5): 0.8767
  - Early stopping triggered at epoch 12
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_best.pth`
  - Mean Test AUC: 0.8311
  - Deney 05'e göre fark: −0.0001
  - Deney 04b'ye göre fark: −0.0031
  - Precision: 0.3616
  - Recall: 0.2355
  - F1 Score: 0.2596
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | 05 Farkı | 04b Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8103 | +0.0017 | +0.0017 |
    | Cardiomegaly | 0.9023 | +0.0022 | +0.0048 |
    | Effusion | 0.8847 | −0.0004 | +0.0025 |
    | Infiltration | 0.7137 | +0.0044 | −0.0029 |
    | Mass | 0.8207 | −0.0034 | −0.0143 |
    | Nodule | 0.7515 | −0.0156 | −0.0005 |
    | Pneumonia | 0.7426 | −0.0068 | −0.0131 |
    | Pneumothorax | 0.8822 | +0.0134 | −0.0012 |
    | Consolidation | 0.8006 | +0.0001 | +0.0012 |
    | Edema | 0.8990 | +0.0031 | −0.0010 |
    | Emphysema | 0.9200 | +0.0001 | −0.0077 |
    | Fibrosis | 0.8089 | +0.0049 | −0.0002 |
    | Pleural_Thickening | 0.7768 | +0.0009 | −0.0047 |
    | Hernia | 0.9221 | −0.0062 | −0.0087 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_clahe_cbam_block34_asl_gn2_gp0_faar_cap3_test_auc_bar.png`
- **Gözlemler:**
  - FAAR cap3 slightly improved Best Val AUC compared with Experiment 04b: 0.8282 vs 0.8275 (+0.0007), but did not improve Mean Test AUC: 0.8311 vs 0.8342 (−0.0031).
  - Compared with Experiment 05, FAAR cap3 produced nearly identical Mean Test AUC: 0.8311 vs 0.8312 (−0.0001), with lower Precision (−0.0177), Recall (−0.0212), and F1 (−0.0192).
  - The cap3 weighting reduced the aggressiveness of the original inverse-frequency FAAR, but this did not translate into better test performance.
  - Our experiment results indicate that FAAR cap3 does not solve the AUC drop observed in Experiment 05.
  - The strongest per-class improvement compared with Experiment 05 was Pneumothorax (+0.0134), while the largest regression was Nodule (−0.0156).
  - Compared with Experiment 04b, most classes regressed, with the largest drops in Mass (−0.0143), Pneumonia (−0.0131), and Hernia (−0.0087).
  - Based on observed validation and test AUC, Experiment 04b remains the strongest ASL-based configuration. Experiment 03b remains the best CLAHE+CBAM BCE model (0.8359), and Experiment 08 is now the best overall Mean Test AUC model (0.8371).
- **Sonraki adım:** improving the best AUC model.

---

### FAAR Tuning Plan

- Experiment 05 showed that FAAR improved Recall and F1 but reduced Mean Test AUC compared with 04b.
- Experiment 05b tested capped inverse-frequency weights (FAAR_CAP_MAX = 3.0) to reduce the aggressiveness of the original inverse-frequency weighting.
- Our experiment results show that both original FAAR (Experiment 05, Mean Test AUC 0.8312) and FAAR cap3 (Experiment 05b, Mean Test AUC 0.8311) did not improve Mean Test AUC compared with Experiment 04b (0.8342).
- FAAR tuning is stopped here. The priority is Mean Test AUC, and further weight-mode variations are not expected to resolve the AUC gap.
- Experiment 06 tested LR=1e-3 on the original BCE baseline — it did not improve AUC and is not continued.
- Experiment 07 tested CBAM block34 without CLAHE. It achieved the best Mean Test AUC so far (0.8361), slightly surpassing Experiment 03b (0.8359). Therefore, CLAHE is not selected for the current best AUC configuration.

---

## [Deney 06] — DenseNet121 + BCE + LR=1e-3 + Patience=10

- **Experiment Name:** `densenet121_bce_lr1e3_pat10`
- **Amaç:**
  - Test whether increasing the learning rate from 1e-4 to 1e-3 improves the original DenseNet121+BCE baseline.
  - All architectural additions (CLAHE, CBAM, ASL, FAAR) were disabled to keep the experiment clean.
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = False
  - USE_CBAM = False
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - Optimizer: Adam
  - Scheduler: ReduceLROnPlateau
  - BATCH_SIZE = 64 | LEARNING_RATE = 1e-3 | NUM_EPOCHS = 50 | PATIENCE = 10 | NUM_WORKERS = 12
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8221
  - Best Epoch: 25
  - Early stopping triggered at epoch 35
  - Val AUC at epoch 35: 0.8207
  - Train Loss at epoch 35: 0.1337
  - Val Loss at epoch 35: 0.1502
  - Val per-class AUC at epoch 35:

    | Sınıf | Val AUC |
    |---|---|
    | Atelectasis | 0.7986 |
    | Cardiomegaly | 0.9048 |
    | Effusion | 0.8842 |
    | Infiltration | 0.7050 |
    | Mass | 0.8330 |
    | Nodule | 0.7355 |
    | Pneumonia | 0.7368 |
    | Pneumothorax | 0.8699 |
    | Consolidation | 0.8043 |
    | Edema | 0.8782 |
    | Emphysema | 0.8873 |
    | Fibrosis | 0.7956 |
    | Pleural_Thickening | 0.8027 |
    | Hernia | 0.8545 |

- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_bce_lr1e3_pat10_best.pth`
  - Mean Test AUC: 0.8247
  - Deney 01'e göre fark: −0.0105
  - Precision: 0.4795
  - Recall: 0.0952
  - F1 Score: 0.1462
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | Deney 01 Farkı |
    |---|---|---|
    | Atelectasis | 0.8005 | −0.0110 |
    | Cardiomegaly | 0.9115 | +0.0125 |
    | Effusion | 0.8863 | +0.0027 |
    | Infiltration | 0.7085 | −0.0022 |
    | Mass | 0.8360 | +0.0024 |
    | Nodule | 0.7413 | −0.0283 |
    | Pneumonia | 0.7450 | −0.0098 |
    | Pneumothorax | 0.8654 | −0.0107 |
    | Consolidation | 0.7977 | −0.0023 |
    | Edema | 0.9016 | −0.0029 |
    | Emphysema | 0.8927 | −0.0265 |
    | Fibrosis | 0.7977 | −0.0233 |
    | Pleural_Thickening | 0.7761 | −0.0084 |
    | Hernia | 0.8859 | −0.0388 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_bce_lr1e3_pat10_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_bce_lr1e3_pat10_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_bce_lr1e3_pat10_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_bce_lr1e3_pat10_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_bce_lr1e3_pat10_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_bce_lr1e3_pat10_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_bce_lr1e3_pat10_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_bce_lr1e3_pat10_test_auc_bar.png`
- **Gözlemler:**
  - Increasing LR from 1e-4 to 1e-3 did not improve performance. Mean Test AUC decreased from 0.8352 to 0.8247 (−0.0105) compared with the original DenseNet121+BCE baseline (Deney 01).
  - Best Val AUC also decreased: 0.8221 vs 0.8319 (−0.0098).
  - Although Precision increased (+0.0569), Recall (−0.0229), F1 (−0.0238), and all AUC metrics declined.
  - Large per-class regressions compared with Deney 01: Hernia (−0.0388), Nodule (−0.0283), Emphysema (−0.0265), Fibrosis (−0.0233).
  - Best epoch shifted to 25 with early stopping at 35 — LR=1e-3 converged later but to a worse optimum.
  - Our experiment results indicate that LR=1e-3 is too aggressive for this dataset and architecture combination.
  - The stronger BCE baseline remains Deney 01 (`densenet121_bce`, Mean Test AUC 0.8352). The best overall model is now Deney 08 (`densenet121_cbam_block34_bce_adamw`, Mean Test AUC 0.8371).
- **Sonraki adım:**

---

## [Deney 07] — DenseNet121 + CBAM block34 + BCE, no CLAHE

- **Experiment Name:** `densenet121_cbam_block34_bce`
- **Amaç:**
  - Test whether removing CLAHE improves the CBAM block34 setup.
  - Direct comparison against Experiment 03b: `densenet121_clahe_cbam_block34_bce`.
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = False
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - BATCH_SIZE = 64 | LEARNING_RATE = 1e-4 | NUM_EPOCHS = 50 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **CBAM Placement:**
  - CBAM after DenseBlock3 before Transition3.
  - CBAM after DenseBlock4 / norm5 / ReLU, before Global Average Pooling.
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8344
  - Best Epoch: 5
  - Train Loss (epoch 5): 0.1382
  - Val Loss (epoch 5): 0.1470
  - Early stopping triggered at epoch 12
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_cbam_block34_bce_best.pth`
  - Mean Test AUC: 0.8361
  - Deney 03b'ye göre fark: +0.0002
  - Deney 01'e göre fark: +0.0009
  - Precision: 0.4421
  - Recall: 0.1284
  - F1 Score: 0.1790
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | 03b Farkı | Baseline Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8138 | +0.0050 | +0.0023 |
    | Cardiomegaly | 0.8969 | +0.0021 | −0.0021 |
    | Effusion | 0.8845 | −0.0013 | +0.0009 |
    | Infiltration | 0.7190 | +0.0061 | +0.0083 |
    | Mass | 0.8393 | +0.0016 | +0.0057 |
    | Nodule | 0.7537 | −0.0057 | −0.0159 |
    | Pneumonia | 0.7487 | −0.0137 | −0.0061 |
    | Pneumothorax | 0.8920 | +0.0064 | +0.0159 |
    | Consolidation | 0.8068 | +0.0101 | +0.0068 |
    | Edema | 0.9016 | +0.0010 | −0.0029 |
    | Emphysema | 0.9193 | −0.0082 | +0.0001 |
    | Fibrosis | 0.8247 | +0.0055 | +0.0037 |
    | Pleural_Thickening | 0.7884 | −0.0022 | +0.0039 |
    | Hernia | 0.9170 | −0.0031 | −0.0077 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_cbam_block34_bce_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_cbam_block34_bce_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_cbam_block34_bce_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_cbam_block34_bce_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_cbam_block34_bce_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_cbam_block34_bce_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_cbam_block34_bce_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_cbam_block34_bce_test_auc_bar.png`
- **Gözlemler:**
  - Removing CLAHE improved Best Val AUC compared with Experiment 03b: 0.8344 vs 0.8320 (+0.0024).
  - Mean Test AUC also slightly improved: 0.8361 vs 0.8359 (+0.0002). At the time of this experiment, Deney 07 was the best overall model by Mean Test AUC.
  - This suggests that CLAHE did not provide a consistent benefit in the CBAM block34 setup, and CBAM block34 works slightly better directly on original X-ray images.
  - Precision (+0.0377), Recall (+0.0234), and F1 (+0.0339) all improved compared with Experiment 03b at threshold 0.5.
  - Per-class improvements compared with 03b: Consolidation (+0.0101), Infiltration (+0.0061), Pneumothorax (+0.0064), Fibrosis (+0.0055), Atelectasis (+0.0050).
  - Per-class regressions compared with 03b: Pneumonia (−0.0137), Emphysema (−0.0082), Nodule (−0.0057), Hernia (−0.0031).
  - Compared with Baseline (Deney 01), the main gains were Pneumothorax (+0.0159), Infiltration (+0.0083), Consolidation (+0.0068), Mass (+0.0057), while Nodule (−0.0159) and Hernia (−0.0077) regressed.
  - Our experiment results show that CLAHE is not necessary for the best AUC performance in this setup. CBAM block34 on original images achieves the best Mean Test AUC.
- **Genel Değerlendirme:**
  - Best Mean Test AUC model: Experiment 08 (0.8371) — CBAM block34 + BCE + AdamW (no CLAHE).
  - Experiment 07 is the best Adam optimizer no-CLAHE CBAM block34 BCE model (0.8361).
  - Experiment 03b remains the best CLAHE+CBAM BCE model (0.8359).
  - Best ASL-based model: Experiment 04b (0.8342) — CLAHE + CBAM block34 + ASL gn2_gp0.
  - AdamW improved Mean Test AUC from 0.8361 to 0.8371, suggesting better generalization in this setup.
- **Sonraki adım:** Experiment 08 tested AdamW on the same no-CLAHE CBAM block34 BCE setup, improving Mean Test AUC from 0.8361 to 0.8371.

---

## [Deney 08] — DenseNet121 + CBAM block34 + BCE + AdamW, no CLAHE

- **Experiment Name:** `densenet121_cbam_block34_bce_adamw`
- **Amaç:**
  - Test whether changing the optimizer from Adam to AdamW improves the current best no-CLAHE CBAM block34 BCE setup.
  - Direct comparison against Experiment 07: `densenet121_cbam_block34_bce`.
- **Config:**
  - Model: DenseNet121
  - Loss: BCEWithLogitsLoss
  - USE_CLAHE = False
  - USE_CBAM = True
  - CBAM_PLACEMENT = "block34"
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
  - Optimizer: AdamW
  - WEIGHT_DECAY = 1e-5
  - BATCH_SIZE = 64 | LEARNING_RATE = 1e-4 | NUM_EPOCHS = 50 | PATIENCE = 7 | NUM_WORKERS = 12 | PIN_MEMORY = True
- **CBAM Placement:**
  - CBAM after DenseBlock3 before Transition3.
  - CBAM after DenseBlock4 / norm5 / ReLU, before Global Average Pooling.
- **Eğitim Sonuçları (Val):**
  - Best Val AUC: 0.8338
  - Best Epoch: 6
  - Train Loss (epoch 6): 0.1345
  - Val Loss (epoch 6): 0.1473
  - Early stopping triggered at epoch 13
- **Test Sonuçları (evaluate.py):**
  - Checkpoint: `results/checkpoints/densenet121_cbam_block34_bce_adamw_best.pth`
  - Mean Test AUC: 0.8371
  - Deney 07'ye göre fark: +0.0010
  - Deney 01'e göre fark: +0.0019
  - Precision: 0.4482
  - Recall: 0.1395
  - F1 Score: 0.1962
  - Threshold: 0.5
  - Per-class Test AUC:

    | Sınıf | Test AUC | Deney 07 Farkı | Baseline Farkı |
    |---|---|---|---|
    | Atelectasis | 0.8169 | +0.0031 | +0.0054 |
    | Cardiomegaly | 0.9026 | +0.0057 | +0.0036 |
    | Effusion | 0.8877 | +0.0032 | +0.0041 |
    | Infiltration | 0.7157 | −0.0033 | +0.0050 |
    | Mass | 0.8452 | +0.0059 | +0.0116 |
    | Nodule | 0.7494 | −0.0043 | −0.0202 |
    | Pneumonia | 0.7595 | +0.0108 | +0.0047 |
    | Pneumothorax | 0.8841 | −0.0079 | +0.0080 |
    | Consolidation | 0.7998 | −0.0070 | −0.0002 |
    | Edema | 0.9063 | +0.0047 | +0.0018 |
    | Emphysema | 0.9214 | +0.0021 | +0.0022 |
    | Fibrosis | 0.8102 | −0.0145 | −0.0108 |
    | Pleural_Thickening | 0.7959 | +0.0075 | +0.0114 |
    | Hernia | 0.9246 | +0.0076 | −0.0001 |

- **Dosyalar:**
  - Loss CSV: `results/metrics/densenet121_cbam_block34_bce_adamw_loss_history.csv`
  - Val AUC CSV: `results/metrics/densenet121_cbam_block34_bce_adamw_val_auc_history.csv`
  - Best Epoch JSON: `results/metrics/densenet121_cbam_block34_bce_adamw_best_epoch.json`
  - Test AUC CSV: `results/metrics/densenet121_cbam_block34_bce_adamw_test_auc_per_class.csv`
  - Test Summary CSV: `results/metrics/densenet121_cbam_block34_bce_adamw_test_summary.csv`
  - Test Metrics JSON: `results/metrics/densenet121_cbam_block34_bce_adamw_test_metrics.json`
  - Curves PNG: `results/figures/densenet121_cbam_block34_bce_adamw_curves.png`
  - Test AUC Bar PNG: `results/figures/densenet121_cbam_block34_bce_adamw_test_auc_bar.png`
- **Gözlemler:**
  - AdamW slightly decreased Best Val AUC compared with Experiment 07: 0.8338 vs 0.8344 (−0.0006).
  - However, AdamW improved Mean Test AUC: 0.8371 vs 0.8361 (+0.0010), making Experiment 08 the best overall model by Mean Test AUC.
  - Precision (+0.0061), Recall (+0.0111), and F1 (+0.0172) all improved compared with Experiment 07 at threshold 0.5.
  - The strongest per-class improvements vs Experiment 07: Pneumonia (+0.0108), Hernia (+0.0076), Pleural_Thickening (+0.0075), Mass (+0.0059), Cardiomegaly (+0.0057).
  - The largest per-class regressions vs Experiment 07: Fibrosis (−0.0145), Pneumothorax (−0.0079), Consolidation (−0.0070), Nodule (−0.0043).
  - Even though validation AUC was slightly lower, the test AUC and threshold metrics improved, suggesting AdamW improved generalization on the test split in this setup.
  - Our experiment results indicate that AdamW with the same weight decay (1e-5) provides a small but consistent improvement in test performance over Adam in the no-CLAHE CBAM block34 BCE setup.
- **Genel Değerlendirme:**
  - Best Mean Test AUC model: Experiment 08 (0.8371) — CBAM block34 + BCE + AdamW (no CLAHE).
  - Experiment 07 is the best Adam optimizer no-CLAHE CBAM block34 BCE model (0.8361).
  - Experiment 03b remains the best CLAHE+CBAM BCE model (0.8359).
  - Best ASL-based model: Experiment 04b (0.8342) — CLAHE + CBAM block34 + ASL gn2_gp0.
  - AdamW improved Mean Test AUC from 0.8361 to 0.8371, suggesting better generalization in this setup.
- **Sonraki adım:** Continue with further improvements, such as testing AdamW on the CLAHE+CBAM block34 setup, or applying ASL with AdamW.

---

## Özet Karşılaştırma Tablosu

| Deney | Config | Best Val AUC | Mean Test AUC | Best Epoch | Infiltration AUC | Pneumonia AUC | Nodule AUC | Pleural_Thickening AUC | Hernia AUC | Emphysema AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 Baseline | bce | 0.8319 | 0.8352 | 6 | 0.7107 | 0.7548 | 0.7696 | 0.7845 | 0.9247 | 0.9192 |
| 02 +CLAHE | clahe_bce | 0.8321 | 0.8339 | 6 | 0.7103 | 0.7456 | 0.7602 | 0.7937 | 0.9060 | 0.9181 |
| 03a +CBAM block4 | clahe_cbam_block4_bce | 0.8301 | 0.8323 | 6 | 0.7152 | 0.7564 | 0.7596 | 0.7804 | 0.9050 | 0.9197 |
| 03b +CBAM block34 | clahe_cbam_block34_bce | 0.8320 | 0.8359 | 5 | 0.7129 | 0.7624 | 0.7594 | 0.7906 | 0.9201 | 0.9275 |
| 03c +CBAM block1234 | clahe_cbam_block1234_bce | 0.8311 | 0.8343 | 5 | 0.7121 | 0.7564 | 0.7638 | 0.7830 | 0.9165 | 0.9211 |
| 04a +ASL (gn4_gp1) | clahe_cbam_block34_asl | 0.8273 | 0.8305 | 6 | 0.7115 | 0.7348 | 0.7478 | 0.7828 | 0.9271 | 0.9288 |
| 04b +ASL (gn2_gp0) | clahe_cbam_block34_asl_gn2_gp0 | 0.8275 | 0.8342 | 5 | 0.7166 | 0.7557 | 0.7520 | 0.7815 | 0.9308 | 0.9277 |
| 05 +FAAR | clahe_cbam_block34_asl_gn2_gp0_faar | 0.8288 | 0.8312 | 5 | 0.7093 | 0.7494 | 0.7671 | 0.7759 | 0.9283 | 0.9199 |
| 05b +FAAR cap3 | clahe_cbam_block34_asl_gn2_gp0_faar_cap3 | 0.8282 | 0.8311 | 5 | 0.7137 | 0.7426 | 0.7515 | 0.7768 | 0.9221 | 0.9200 |
| 06 BCE LR=1e-3 | bce_lr1e3_pat10 | 0.8221 | 0.8247 | 25 | 0.7085 | 0.7450 | 0.7413 | 0.7761 | 0.8859 | 0.8927 |
| 07 CBAM block34 no CLAHE | cbam_block34_bce | 0.8344 | 0.8361 | 5 | 0.7190 | 0.7487 | 0.7537 | 0.7884 | 0.9170 | 0.9193 |
| 08 CBAM block34 no CLAHE + AdamW | cbam_block34_bce_adamw | 0.8338 | 0.8371 | 6 | 0.7157 | 0.7595 | 0.7494 | 0.7959 | 0.9246 | 0.9214 |
