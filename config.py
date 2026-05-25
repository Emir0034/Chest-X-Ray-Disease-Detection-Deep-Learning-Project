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
BATCH_SIZE    = 64 # default 32
LEARNING_RATE = 2e-5 # normalde 1e-4
NUM_EPOCHS    = 20 # 50
PATIENCE      = 5       # early stopping patience (epochs without val AUC improvement) | normalde 7 
NUM_WORKERS   = 12      # default 4 but 0 for more safety starting (şimdilik)
PIN_MEMORY    = True
OPTIMIZER_NAME    = "adamw"  # "adam" or "adamw"
WEIGHT_DECAY      = 1e-5    # normalde 1e-5
FINE_TUNE_FROM_CHECKPOINT = "results/checkpoints/densenet121_cbam_block34_bce_adamw_best.pth"  # empty = start from scratch / ImageNet-pretrained
                                # set to a checkpoint path to initialize model weights only
                                # (optimizer, scheduler, epoch, best AUC are NOT resumed)
                                # example: "results/checkpoints/densenet121_cbam_block34_bce_adamw_best.pth"
EXPERIMENT_SUFFIX  = "stage2_from_exp08_asl_lr2e5"       # optional suffix, e.g. "wd1e4" or "lr5e5"
# Use "trial" for exploratory runs that should not be treated as official experiments.
# Use "official" for final documented experiments.
EXPERIMENT_STATUS  = "trial"  # "official" or "trial"
if EXPERIMENT_STATUS not in ("official", "trial"):
    raise ValueError(
        f"EXPERIMENT_STATUS must be 'official' or 'trial', got {EXPERIMENT_STATUS!r}."
    )

# ── Dataset paths  (edit these before running) ───────────────────────────────
DATA_CSV   = r"data/NIH Chest X-rays/Data_Entry_2017.csv"   # NIH ChestX-ray14 labels CSV
IMAGES_DIR = r"data/NIH Chest X-rays"               # flat directory containing all PNG files
SPLIT_DIR  = r"splits"             # where train/val/test CSV splits are saved

# ── Experiment flags ─────────────────────────────────────────────────────────
# Toggle these to select which experiment to run.
USE_CLAHE           = False    # Experiment 2+: CLAHE contrast enhancement
USE_CBAM            = True  # Experiment 3+: CBAM attention module
CBAM_PLACEMENT      = "block34"   # "block4" | "block34" | "block1234" — ignored when USE_CBAM=False
USE_ASYMMETRIC_LOSS = True  # Experiment 4:  Asymmetric Loss (replaces BCE)
ASL_GAMMA_NEG   = 2.0        # Focusing parameter for negative samples
ASL_GAMMA_POS   = 0.0        # Focusing parameter for positive samples
ASL_CLIP        = 0.05       # Probability margin for easy negative suppression
USE_FOCAL_LOSS  = False       # Experiment 10: Focal Loss (replaces BCE when ASL is off)
FOCAL_GAMMA     = 2.0        # Focusing parameter for Focal Loss
FOCAL_ALPHA     = None       # Optional scalar alpha weight (None = no alpha weighting)
USE_FAAR        = False     # Experiment 5+: Frequency-Aware Attention Refinement
FAAR_ALPHA_INIT = 0.0        # Initial residual scale (0.0 → identity at step 0)

# ── FAAR tuning settings ─────────────────────────────────────────────────────
# Options:
#   "inverse"  : original inverse-frequency FAAR
#   "cap"      : inverse-frequency FAAR with max cap
#   "sqrt_inv" : softer square-root inverse-frequency FAAR
#
# For Experiment 05b:
FAAR_WEIGHT_MODE = "cap"
FAAR_CAP_MAX     = 3.0

# ── Results directories ──────────────────────────────────────────────────────
RESULTS_DIR = "results"

# Official mode uses results/ directly.
# Trial mode routes all outputs under results/trials/ to keep them isolated.
_RESULTS_SUBDIR = os.path.join(RESULTS_DIR, "trials") if EXPERIMENT_STATUS == "trial" else RESULTS_DIR

CHECKPOINTS_DIR = os.path.join(_RESULTS_SUBDIR, "checkpoints")
METRICS_DIR     = os.path.join(_RESULTS_SUBDIR, "metrics")
FIGURES_DIR     = os.path.join(_RESULTS_SUBDIR, "figures")
GRADCAM_DIR     = os.path.join(_RESULTS_SUBDIR, "figures", "gradcam")

# Always points to official checkpoints regardless of EXPERIMENT_STATUS.
# Used by gradcam_compare.py which always loads Exp08/09 official checkpoints.
OFFICIAL_CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")


def get_experiment_name() -> str:
    #Return a short tag describing the current experiment configuration.
    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else str(v)

    parts = ["densenet121"]
    if USE_CLAHE:
        parts.append("clahe")
    if USE_CBAM:
        parts.append("cbam")
        parts.append(CBAM_PLACEMENT)   # e.g. "block4", "block34", "block1234"
    if USE_ASYMMETRIC_LOSS:
        parts.append(f"asl_gn{_fmt(ASL_GAMMA_NEG)}_gp{_fmt(ASL_GAMMA_POS)}")
    elif USE_FOCAL_LOSS:
        parts.append(f"focal_gamma{_fmt(FOCAL_GAMMA)}")
    else:
        parts.append("bce")
    if USE_FAAR:
        if FAAR_WEIGHT_MODE == "inverse":
            parts.append("faar")
        elif FAAR_WEIGHT_MODE == "cap":
            parts.append(f"faar_cap{_fmt(FAAR_CAP_MAX)}")
        elif FAAR_WEIGHT_MODE == "sqrt_inv":
            parts.append("faar_sqrtinv")
        else:
            raise ValueError(f"Unknown FAAR_WEIGHT_MODE: {FAAR_WEIGHT_MODE!r}")
    if OPTIMIZER_NAME.lower() == "adamw":
        parts.append("adamw")
    if EXPERIMENT_SUFFIX:
        parts.append(EXPERIMENT_SUFFIX)
    if EXPERIMENT_STATUS == "trial":
        parts.append("trial")
    return "_".join(parts)


"""
EXPERIMENT_NAME_OVERRIDE = "densenet121_bce_lr1e3_pat10"
def get_experiment_name() -> str:
    #Return a short tag describing the current experiment configuration

    if EXPERIMENT_NAME_OVERRIDE:
        return EXPERIMENT_NAME_OVERRIDE

    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else str(v)

    parts = ["densenet121"]
    if USE_CLAHE:
        parts.append("clahe")
    if USE_CBAM:
        parts.append("cbam")
        parts.append(CBAM_PLACEMENT)
    if USE_ASYMMETRIC_LOSS:
        parts.append(f"asl_gn{_fmt(ASL_GAMMA_NEG)}_gp{_fmt(ASL_GAMMA_POS)}")
    elif USE_FOCAL_LOSS:
        parts.append(f"focal_gamma{_fmt(FOCAL_GAMMA)}")
    else:
        parts.append("bce")
    if USE_FAAR:
        if FAAR_WEIGHT_MODE == "inverse":
            parts.append("faar")
        elif FAAR_WEIGHT_MODE == "cap":
            parts.append(f"faar_cap{_fmt(FAAR_CAP_MAX)}")
        elif FAAR_WEIGHT_MODE == "sqrt_inv":
            parts.append("faar_sqrtinv")
        else:
            raise ValueError(f"Unknown FAAR_WEIGHT_MODE: {FAAR_WEIGHT_MODE!r}")
    return "_".join(parts)
"""