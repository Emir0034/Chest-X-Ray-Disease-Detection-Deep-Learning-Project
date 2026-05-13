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
  - USE_CLAHE = True
  - USE_CBAM = False
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
- **Eğitim Sonuçları (Val):**
  - Best Val AUC:
  - Best Epoch:
  - Train Loss:
  - Val Loss:
- **Test Sonuçları (evaluate.py):**
  - Mean Test AUC:
  - Deney 01'e göre fark:
  - Per-class Test AUC:
    | Sınıf | Test AUC | Deney 01 Farkı |
    |---|---|---|
    | Atelectasis | | |
    | Cardiomegaly | | |
    | Effusion | | |
    | Infiltration | | |
    | Mass | | |
    | Nodule | | |
    | Pneumonia | | |
    | Pneumothorax | | |
    | Consolidation | | |
    | Edema | | |
    | Emphysema | | |
    | Fibrosis | | |
    | Pleural_Thickening | | |
    | Hernia | | |
- **Dosyalar:**
  - Loss CSV:
  - Val AUC CSV:
  - Test AUC CSV:
  - Curves PNG:
  - Test AUC Bar PNG:
- **Gözlemler:**
- **Sonraki adım:**

---

## [Deney 03] — +CLAHE +CBAM
- **Experiment Name:** `densenet121_clahe_cbam_bce`
- **Config:**
  - USE_CLAHE = True
  - USE_CBAM = True
  - USE_ASYMMETRIC_LOSS = False
  - USE_FAAR = False
- **Eğitim Sonuçları (Val):**
  - Best Val AUC:
  - Best Epoch:
  - Train Loss:
  - Val Loss:
- **Test Sonuçları (evaluate.py):**
  - Mean Test AUC:
  - Deney 02'ye göre fark:
  - Per-class Test AUC:
    | Sınıf | Test AUC | Deney 02 Farkı |
    |---|---|---|
    | Atelectasis | | |
    | Cardiomegaly | | |
    | Effusion | | |
    | Infiltration | | |
    | Mass | | |
    | Nodule | | |
    | Pneumonia | | |
    | Pneumothorax | | |
    | Consolidation | | |
    | Edema | | |
    | Emphysema | | |
    | Fibrosis | | |
    | Pleural_Thickening | | |
    | Hernia | | |
- **Dosyalar:**
  - Loss CSV:
  - Val AUC CSV:
  - Test AUC CSV:
  - Curves PNG:
  - Test AUC Bar PNG:
- **Gözlemler:**
- **Sonraki adım:**

---

## [Deney 04] — +CLAHE +CBAM +ASL
- **Experiment Name:** `densenet121_clahe_cbam_asl`
- **Config:**
  - USE_CLAHE = True
  - USE_CBAM = True
  - USE_ASYMMETRIC_LOSS = True
  - USE_FAAR = False
- **Eğitim Sonuçları (Val):**
  - Best Val AUC:
  - Best Epoch:
  - Train Loss:
  - Val Loss:
- **Test Sonuçları (evaluate.py):**
  - Mean Test AUC:
  - Deney 03'e göre fark:
  - Per-class Test AUC:
    | Sınıf | Test AUC | Deney 03 Farkı |
    |---|---|---|
    | Atelectasis | | |
    | Cardiomegaly | | |
    | Effusion | | |
    | Infiltration | | |
    | Mass | | |
    | Nodule | | |
    | Pneumonia | | |
    | Pneumothorax | | |
    | Consolidation | | |
    | Edema | | |
    | Emphysema | | |
    | Fibrosis | | |
    | Pleural_Thickening | | |
    | Hernia | | |
- **Dosyalar:**
  - Loss CSV:
  - Val AUC CSV:
  - Test AUC CSV:
  - Curves PNG:
  - Test AUC Bar PNG:
- **Gözlemler:**
- **Sonraki adım:**

---

## [Deney 05] — +CLAHE +CBAM +ASL +FAAR (Final)
- **Experiment Name:** `densenet121_clahe_cbam_asl_faar`
- **Config:**
  - USE_CLAHE = True
  - USE_CBAM = True
  - USE_ASYMMETRIC_LOSS = True
  - USE_FAAR = True
- **Eğitim Sonuçları (Val):**
  - Best Val AUC:
  - Best Epoch:
  - Train Loss:
  - Val Loss:
- **Test Sonuçları (evaluate.py):**
  - Mean Test AUC:
  - Deney 04'e göre fark:
  - Baseline'a göre toplam fark:
  - Per-class Test AUC:
    | Sınıf | Test AUC | Deney 04 Farkı | Baseline Farkı |
    |---|---|---|---|
    | Atelectasis | | | |
    | Cardiomegaly | | | |
    | Effusion | | | |
    | Infiltration | | | |
    | Mass | | | |
    | Nodule | | | |
    | Pneumonia | | | |
    | Pneumothorax | | | |
    | Consolidation | | | |
    | Edema | | | |
    | Emphysema | | | |
    | Fibrosis | | | |
    | Pleural_Thickening | | | |
    | Hernia | | | |
- **Dosyalar:**
  - Loss CSV:
  - Val AUC CSV:
  - Test AUC CSV:
  - Curves PNG:
  - Test AUC Bar PNG:
  - Grad-CAM:
- **Gözlemler:**
- **Genel Değerlendirme:**

---

## Özet Karşılaştırma Tablosu (Test AUC)

| Deney | Config | Mean Test AUC | Best Epoch | Hernia AUC | Emphysema AUC |
|---|---|---|---|---|---|
| 01 Baseline | bce | 0.8352 | 6 | 0.9247 | 0.9192 |
| 02 +CLAHE | clahe_bce | | | | |
| 03 +CBAM | clahe_cbam_bce | | | | |
| 04 +ASL | clahe_cbam_asl | | | | |
| 05 +FAAR | clahe_cbam_asl_faar | | | | |
