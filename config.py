import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Input / model ────────────────────────────────────────────────────────────
IMAGE_SIZE  = 224
NUM_CLASSES = 14

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE    = 4 # default 32
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 1 # 50
PATIENCE      = 5       # early stopping patience (epochs without val AUC improvement)
NUM_WORKERS   = 0       # default 4 but 0 for more safety starting (şimdilik)
PIN_MEMORY    = True

# ── Dataset paths  (edit these before running) ───────────────────────────────
DATA_CSV   = r"data/NIH Chest X-rays/Data_Entry_2017.csv"   # NIH ChestX-ray14 labels CSV
IMAGES_DIR = r"data/NIH Chest X-rays"               # flat directory containing all PNG files
SPLIT_DIR  = r"splits"             # where train/val/test CSV splits are saved

# ── Experiment flags ─────────────────────────────────────────────────────────
# Toggle these to select which experiment to run.
USE_CLAHE           = False   # Experiment 2+: CLAHE contrast enhancement
USE_CBAM            = False   # Experiment 3+: CBAM attention module
USE_ASYMMETRIC_LOSS = False   # Experiment 4:  Asymmetric Loss (replaces BCE)

# ── Results directories ──────────────────────────────────────────────────────
RESULTS_DIR     = "results"
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
GRADCAM_DIR     = os.path.join(RESULTS_DIR, "figures", "gradcam")


def get_experiment_name() -> str:
    """Return a short tag describing the current experiment configuration."""
    parts = ["densenet121"]
    if USE_CLAHE:
        parts.append("clahe")
    if USE_CBAM:
        parts.append("cbam")
    parts.append("asl" if USE_ASYMMETRIC_LOSS else "bce")
    return "_".join(parts)
