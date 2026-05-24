"""
gradcam_compare.py — Side-by-side Grad-CAM comparison for Experiment 08 and Experiment 09.

Experiment 08: densenet121_cbam_block34_bce_adamw  (best mean AUC, 0.8371)
Experiment 09: densenet121_cbam_block34_asl_gn2_gp0_adamw  (best Recall/F1)

Both use the same architecture: DenseNet121 + CBAM block34, no CLAHE, no FAAR.
Only checkpoint weights differ.

Usage:
    python gradcam_compare.py --image path/to/image.png --class_name Effusion
    python gradcam_compare.py --image path/to/image.png --class_name Effusion --output_name sample_effusion
    python gradcam_compare.py --find_examples --class_name Effusion --max_examples 5
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from dataset import _build_path_index, create_patient_splits
from gradcam import GradCAM, overlay_heatmap
from model import ChestXRayModel
from transforms import get_val_transforms


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Experiment definitions ────────────────────────────────────────────────────
EXP08_CKPT  = os.path.join(config.OFFICIAL_CHECKPOINTS_DIR, "densenet121_cbam_block34_bce_adamw_best.pth")
EXP09_CKPT  = os.path.join(config.OFFICIAL_CHECKPOINTS_DIR, "densenet121_cbam_block34_asl_gn2_gp0_adamw_best.pth")
EXP08_LABEL = "Exp08 BCE+AdamW"
EXP09_LABEL = "Exp09 ASL+AdamW"

# ── Class names (must match config.DISEASE_LABELS order) ─────────────────────
CLASS_NAMES = [
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


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model() -> ChestXRayModel:
    """Build the shared Exp08/09 architecture without loading pretrained ImageNet weights.

    pretrained=False because checkpoint weights overwrite parameters immediately.
    Architecture is identical for both experiments — only loss during training differed.
    """
    return ChestXRayModel(
        num_classes=14,
        use_cbam=True,
        cbam_placement="block34",
        pretrained=False,
        use_faar=False,
    )


def _load_weights(ckpt_path: str, model: ChestXRayModel) -> dict:
    """Load checkpoint weights into model. Tries multiple key conventions robustly.

    Returns the checkpoint metadata dict (epoch, val_auc, etc.) for display.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        meta = {k: v for k, v in state.items() if k != "model_state_dict"}
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
        meta = {k: v for k, v in state.items() if k != "state_dict"}
    else:
        model.load_state_dict(state)
        meta = {}
    epoch    = meta.get("epoch", "?")
    val_auc  = meta.get("val_auc", float("nan"))
    val_str  = f"{val_auc:.4f}" if isinstance(val_auc, float) else str(val_auc)
    print(f"  Loaded: {ckpt_path}")
    print(f"          epoch={epoch}  val_auc={val_str}")
    return meta


def _find_target_layer(model: ChestXRayModel):
    """Locate and return the Grad-CAM target layer, printing which one is used.

    Checks the two most common attribute paths for this project's model wrapper.
    This function is for diagnostics only — GradCAM hooks directly on
    model.features.denseblock4 internally.
    """
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        print("  Grad-CAM target layer: model.features.denseblock4")
        return model.features.denseblock4
    if (hasattr(model, "backbone")
            and hasattr(model.backbone, "features")
            and hasattr(model.backbone.features, "denseblock4")):
        print("  Grad-CAM target layer: model.backbone.features.denseblock4")
        return model.backbone.features.denseblock4
    raise RuntimeError(
        "Cannot locate denseblock4. "
        "Checked: model.features.denseblock4 and model.backbone.features.denseblock4. "
        "Inspect model.py and update _find_target_layer() accordingly."
    )


# ── Image helpers ─────────────────────────────────────────────────────────────

def _load_image(image_path: str):
    """Load and preprocess an image for inference and display.

    Returns:
        input_tensor: shape (1, 3, 224, 224), normalized, on DEVICE.
        display_np:   uint8 RGB ndarray (224, 224, 3) for matplotlib.
    """
    raw = Image.open(image_path).convert("RGB")
    display_np = np.array(raw.resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    val_transforms = get_val_transforms(use_clahe=False, image_size=config.IMAGE_SIZE)
    input_tensor = val_transforms(raw).unsqueeze(0).to(DEVICE)
    return input_tensor, display_np


# ── Grad-CAM runner ───────────────────────────────────────────────────────────

def _run_gradcam(model: ChestXRayModel, input_tensor: torch.Tensor, class_idx: int):
    """Run Grad-CAM for one model and return (heatmap, sigmoid_probability).

    A separate no-grad forward pass gets the probability first;
    then GradCAM runs a fresh forward+backward pass for the gradient maps.

    Returns:
        heatmap: float32 ndarray (224, 224), values in [0, 1].
        prob:    sigmoid probability for the target class (float).
    """
    model.eval()
    with torch.no_grad():
        logit = model(input_tensor)[0, class_idx]
        prob = torch.sigmoid(logit).item()
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(input_tensor, class_idx)
    gradcam.remove_hooks()
    return heatmap, prob


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_image(array_rgb: np.ndarray, path: str) -> None:
    """Save a uint8 RGB ndarray as a PNG using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(array_rgb)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_comparison(
    display_np: np.ndarray,
    overlay08: np.ndarray,
    overlay09: np.ndarray,
    class_name: str,
    prob08: float,
    prob09: float,
    save_path: str,
) -> None:
    """Save a 1×3 comparison figure: Original | Exp08 overlay | Exp09 overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(display_np, cmap="gray")
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay08)
    axes[1].set_title(f"{EXP08_LABEL}\n{class_name}  p={prob08:.3f}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay09)
    axes[2].set_title(f"{EXP09_LABEL}\n{class_name}  p={prob09:.3f}", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(f"Grad-CAM Comparison: {class_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Find-examples mode ────────────────────────────────────────────────────────

def _find_examples(class_name: str, max_examples: int) -> None:
    """Print file paths of positive test-set images for a given disease class.

    Uses the persisted patient-level test split (splits/test.csv).
    If the split CSV does not exist yet, create_patient_splits() will
    generate it from DATA_CSV using the same seed as training.
    """
    print(f"\nSearching test split for positive '{class_name}' examples...\n")
    _, _, test_df = create_patient_splits(
        csv_path  = config.DATA_CSV,
        split_dir = config.SPLIT_DIR,
        seed      = config.SEED,
    )
    positives = test_df[
        test_df["Finding Labels"].str.contains(class_name, na=False, regex=False)
    ]
    if positives.empty:
        print(f"No positive test samples found for '{class_name}'.")
        return

    path_index = _build_path_index(config.IMAGES_DIR)
    count = 0
    for _, row in positives.iterrows():
        img_path = path_index.get(row["Image Index"])
        if img_path is not None:
            print(img_path)
            count += 1
        if count >= max_examples:
            break

    if count == 0:
        print(
            f"Found {len(positives)} positive rows in the test split, "
            "but none of their image files were located on disk.\n"
            f"Check that config.IMAGES_DIR='{config.IMAGES_DIR}' is correct."
        )
    else:
        print(f"\n{count} example(s) listed above. "
              "Use --image <path> to generate Grad-CAM for one of them.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grad-CAM comparison between Experiment 08 (BCE) and Experiment 09 (ASL)."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to the chest X-ray image to visualize.",
    )
    parser.add_argument(
        "--class_name", type=str, required=True,
        help=f"Target disease class. One of: {', '.join(CLASS_NAMES)}",
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Base name for output files (without extension). "
             "Defaults to <image_stem>_<class_name>.",
    )
    parser.add_argument(
        "--find_examples", action="store_true",
        help="Print candidate positive test images for --class_name and exit.",
    )
    parser.add_argument(
        "--max_examples", type=int, default=5,
        help="Maximum number of example paths to print (default: 5).",
    )
    args = parser.parse_args()

    # ── Validate class name ───────────────────────────────────────────────────
    if args.class_name not in CLASS_NAMES:
        print(
            f"ERROR: '{args.class_name}' is not a valid class name.\n"
            f"Valid options: {', '.join(CLASS_NAMES)}"
        )
        sys.exit(1)
    class_idx = CLASS_NAMES.index(args.class_name)

    # ── Find-examples mode ────────────────────────────────────────────────────
    if args.find_examples:
        _find_examples(args.class_name, args.max_examples)
        return

    # ── Require --image for comparison mode ───────────────────────────────────
    if args.image is None:
        parser.error("--image is required unless --find_examples is set.")

    if not os.path.isfile(args.image):
        print(f"ERROR: Image file not found: {args.image}")
        sys.exit(1)

    # ── Resolve output name ───────────────────────────────────────────────────
    if args.output_name:
        output_name = args.output_name
    else:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        output_name = f"{stem}_{args.class_name}"

    os.makedirs(config.GRADCAM_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Grad-CAM Comparison")
    print(f"Device     : {DEVICE}")
    print(f"Class      : {args.class_name}  (index {class_idx})")
    print(f"Image      : {args.image}")
    print(f"Output dir : {config.GRADCAM_DIR}")
    print(f"Output name: {output_name}")
    print(f"{'='*60}\n")

    # ── Load image ────────────────────────────────────────────────────────────
    print("Loading image...")
    input_tensor, display_np = _load_image(args.image)

    # ── Load Experiment 08 ────────────────────────────────────────────────────
    print("\n--- Experiment 08 (BCE + AdamW) ---")
    model08 = _build_model().to(DEVICE)
    _load_weights(EXP08_CKPT, model08)
    _find_target_layer(model08)
    heatmap08, prob08 = _run_gradcam(model08, input_tensor, class_idx)
    print(f"  {args.class_name} probability: {prob08:.4f}")
    overlay08 = overlay_heatmap(display_np, heatmap08, alpha=0.4)

    # ── Load Experiment 09 ────────────────────────────────────────────────────
    print("\n--- Experiment 09 (ASL + AdamW) ---")
    model09 = _build_model().to(DEVICE)
    _load_weights(EXP09_CKPT, model09)
    _find_target_layer(model09)
    heatmap09, prob09 = _run_gradcam(model09, input_tensor, class_idx)
    print(f"  {args.class_name} probability: {prob09:.4f}")
    overlay09 = overlay_heatmap(display_np, heatmap09, alpha=0.4)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    _save_image(display_np,  os.path.join(config.GRADCAM_DIR, f"{output_name}_original.png"))
    _save_image(overlay08,   os.path.join(config.GRADCAM_DIR, f"{output_name}_exp08_overlay.png"))
    _save_image(overlay09,   os.path.join(config.GRADCAM_DIR, f"{output_name}_exp09_overlay.png"))
    _save_comparison(
        display_np, overlay08, overlay09,
        args.class_name, prob08, prob09,
        os.path.join(config.GRADCAM_DIR, f"{output_name}_comparison.png"),
    )

    print(f"\nDone. All files saved to: {config.GRADCAM_DIR}")


if __name__ == "__main__":
    main()
