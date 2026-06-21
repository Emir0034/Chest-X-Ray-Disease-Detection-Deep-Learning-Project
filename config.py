import os

SEED = 42

IMAGE_SIZE  = 224
NUM_CLASSES = 14

# All diseases
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

# Training hyperparameters
BATCH_SIZE    = 64 # default 32
LEARNING_RATE = 1e-4 # normalde 1e-4
NUM_EPOCHS    = 50 # 50
PATIENCE      = 7 # early stopping patience | normalde 7
NUM_WORKERS   = 12 # default 4 but 0 for more safety starting (şimdilik)
PIN_MEMORY    = True
OPTIMIZER_NAME    = "adamw"   # "adam" or "adamw" | normalde adam
WEIGHT_DECAY      = 1e-5      # normalde 1e-5
SCHEDULER_NAME    = "plateau"   # "plateau" or "cosine" | "plateau" is the default one
COSINE_T_MAX      = 20         # CosineAnnealingLR: number of epochs per cycle
COSINE_ETA_MIN    = 1e-6       # CosineAnnealingLR: minimum learning rate
FINE_TUNE_FROM_CHECKPOINT = ""  # empty = start from scratch / ImageNet-pretrained
                                # set to a checkpoint path to initialize model weights only
                                # example: "results/checkpoints/densenet121_cbam_block34_bce_adamw_best.pth"
EXPERIMENT_SUFFIX  = "" # optional suffix, e.g. "wd1e4" or "lr5e5"
# Use "trial" for exploratory runs that should not be treated as official experiments.
# Use "official" for final documented experiments.
EXPERIMENT_STATUS  = "official"  # "official" or "trial"
if EXPERIMENT_STATUS not in ("official", "trial"):
    raise ValueError(
        f"EXPERIMENT_STATUS must be 'official' or 'trial', got {EXPERIMENT_STATUS!r}."
    )

# Dataset paths
DATA_CSV   = r"data/NIH Chest X-rays/Data_Entry_2017.csv" # NIH ChestX-ray14 labels csv
# flat directory containing all png files
IMAGES_DIR = r"data/NIH Chest X-rays"
# where train/val/test csv splits are saved
SPLIT_DIR  = r"splits"

# Split mode: "custom" = legacy 70/15/15 patient-level split (splits/*.csv)
#             "official_nih" = official NIH train_val_list.txt / test_list.txt,
#                               with train_val split patient-level 90/10 into train/val
SPLIT_MODE = "official_nih"

# Official NIH ChestX-ray14 split files
NIH_TRAIN_VAL_LIST = r"data/NIH Chest X-rays/train_val_list.txt"
NIH_TEST_LIST      = r"data/NIH Chest X-rays/test_list.txt"
OFFICIAL_SPLIT_DIR = r"splits/official_nih"
OFFICIAL_VAL_RATIO = 0.10   # patient-level 90/10 split of train_val_list.txt


USE_CLAHE           = False # Experiment 2+: CLAHE contrast enhancement
USE_CBAM            = True # Experiment 3+: CBAM attention module
CBAM_PLACEMENT      = "block34"   # "block4" | "block34" | "block1234" ignored when USE_CBAM=False
USE_ASYMMETRIC_LOSS = False # Experiment 4: Asymmetric Loss (replaces BCE)
ASL_GAMMA_NEG   = 2.0 # Focusing parameter for negative samples
ASL_GAMMA_POS   = 0.0 # Focusing parameter for positive samples
ASL_CLIP        = 0.05 # Probability margin for easy negative suppression
USE_FOCAL_LOSS  = False # Experiment 10: Focal Loss (replaces BCE when ASL is off)
FOCAL_GAMMA     = 2.0 
FOCAL_ALPHA     = None # Optional scalar alpha weight (None = no alpha weighting)
USE_FAAR        = False # Experiment 5+: Frequency Aware Attention Refinement
FAAR_ALPHA_INIT = 0.0 # Initial residual scale 


# FAAR settings 
# "cap" : inverse-frequency FAAR with max cap
# For Experiment 05b:
FAAR_WEIGHT_MODE = "cap"
FAAR_CAP_MAX     = 3.0


RESULTS_DIR = "results"

# Official mod sonuçları doğrudan results/ klasörüne kaydeder
# Trial mode routes all outputs under results/trials/ to keep them isolated
_RESULTS_SUBDIR = os.path.join(RESULTS_DIR, "trials") if EXPERIMENT_STATUS == "trial" else RESULTS_DIR

CHECKPOINTS_DIR = os.path.join(_RESULTS_SUBDIR, "checkpoints")
METRICS_DIR     = os.path.join(_RESULTS_SUBDIR, "metrics")
FIGURES_DIR     = os.path.join(_RESULTS_SUBDIR, "figures")
GRADCAM_DIR     = os.path.join(_RESULTS_SUBDIR, "figures", "gradcam")

# trial olsa bile burası hep official checkpoint e bakar
OFFICIAL_CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")


def get_experiment_name() -> str:
    # Build experiment name 
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
    if OPTIMIZER_NAME.lower() == "adamw":
        parts.append("adamw")
    if SPLIT_MODE == "official_nih":
        parts.append("official_split")
    if EXPERIMENT_SUFFIX:
        parts.append(EXPERIMENT_SUFFIX)
    if EXPERIMENT_STATUS == "trial":
        parts.append("trial")
    return "_".join(parts)


