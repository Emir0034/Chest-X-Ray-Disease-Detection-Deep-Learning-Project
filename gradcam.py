import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
from dataset import _build_path_index, create_patient_splits
from model import build_model
from transforms import get_val_transforms
from utils import load_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ChestXRayModel.

    Hooks are registered on model.features.denseblock4, which is the last
    dense block of DenseNet121 and produces the richest spatial feature map
    before the norm/relu/pooling stages.

    The forward hook in model.py uses F.relu(inplace=False), which is required
    so that backward hooks on denseblock4 do not encounter in-place modified
    tensors (which would raise RuntimeError during backward).

    Usage:
        cam = GradCAM(model)
        heatmap = cam.generate(input_tensor, class_idx)
        cam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.model.eval()
        self._activations = {}
        self._gradients   = {}

        self._fwd_handle = model.features.denseblock4.register_forward_hook(
            self._save_activation
        )
        self._bwd_handle = model.features.denseblock4.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self._activations["value"] = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients["value"] = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate a Grad-CAM heatmap for one image and one class.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, 3, H, W), on device.
            class_idx:    Index of the target disease class (0–13).

        Returns:
            Normalised heatmap as float32 ndarray, shape (224, 224), values in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)          # (1, 14)
        logits[0, class_idx].backward()

        acts   = self._activations["value"]        # (1, C, H, W)
        grads  = self._gradients["value"]          # (1, C, H, W)

        # Global-average-pool the gradients over spatial dims → importance weights
        weights = grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)

        # Upsample to input resolution and normalise to [0, 1]
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()                  # (224, 224)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks to free memory."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(
    image_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto an original image.

    Args:
        image_np: Original image as uint8 RGB ndarray, shape (H, W, 3).
        cam:      Normalised heatmap, shape (H, W), values in [0, 1].
        alpha:    Weight for the heatmap overlay (default 0.4).

    Returns:
        Blended image as uint8 RGB ndarray, shape (H, W, 3).
    """
    heatmap_uint8 = (cam * 255).astype(np.uint8)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # BGR
    image_bgr     = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    blended_bgr   = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)


def generate_gradcam_for_sample(
    model: torch.nn.Module,
    image_path: str,
    class_idx: int,
    class_name: str,
    exp_name: str,
    val_transforms,
    device: torch.device,
    save_dir: str,
) -> None:
    """Generate and save a side-by-side Grad-CAM figure for one image.

    The figure has two panels: original resized image (left) and
    heatmap overlay (right). Saved as PNG under save_dir.
    """
    # Load and preprocess
    raw_image = Image.open(image_path).convert("RGB")
    raw_np    = np.array(raw_image.resize((224, 224), Image.BILINEAR), dtype=np.uint8)

    input_tensor = val_transforms(raw_image).unsqueeze(0).to(device)

    # Generate heatmap
    gradcam = GradCAM(model)
    cam     = gradcam.generate(input_tensor, class_idx)
    gradcam.remove_hooks()

    overlay = overlay_heatmap(raw_np, cam, alpha=0.4)

    # Save side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(raw_np, cmap="gray")
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(overlay)
    ax2.set_title(f"Grad-CAM: {class_name}")
    ax2.axis("off")

    fig.suptitle(f"{exp_name} | {class_name}", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{exp_name}_{class_name}_gradcam.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main() -> None:
    """Generate Grad-CAM heatmaps for one positive test image per disease class.

    Loads the best checkpoint for the current experiment (set flags in config.py).
    For each of the 14 disease classes, finds the first test image that has that
    disease label and generates a heatmap. Images that have no positive test sample
    for a class are skipped with a warning.
    """
    exp_name = config.get_experiment_name()
    print(f"\n{'='*60}")
    print(f"Grad-CAM   : {exp_name}")
    print(f"Device     : {DEVICE}")
    print(f"{'='*60}\n")

    os.makedirs(config.GRADCAM_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    best_ckpt = os.path.join(config.CHECKPOINTS_DIR, f"{exp_name}_best.pth")
    model = build_model(config).to(DEVICE)
    ckpt  = load_checkpoint(best_ckpt, model)
    model.eval()
    print(f"Loaded: {best_ckpt}  (epoch {ckpt.get('epoch','?')}, val AUC {ckpt.get('val_auc',0):.4f})\n")

    val_transforms = get_val_transforms(use_clahe=config.USE_CLAHE, image_size=config.IMAGE_SIZE)

    # ── Build image path index ─────────────────────────────────────────────────
    path_index = _build_path_index(config.IMAGES_DIR)

    # ── Load test split and find positive samples ─────────────────────────────
    _, _, test_df = create_patient_splits(
        csv_path  = config.DATA_CSV,
        split_dir = config.SPLIT_DIR,
        seed      = config.SEED,
    )

    for class_idx, class_name in enumerate(config.DISEASE_LABELS):
        # Find test rows that contain this class in their Finding Labels
        positive_rows = test_df[
            test_df["Finding Labels"].str.contains(class_name, na=False)
        ]

        if positive_rows.empty:
            print(f"  [{class_name}] No positive test samples found — skipping.")
            continue

        # Pick the first positive image
        filename  = positive_rows.iloc[0]["Image Index"]
        img_path  = path_index.get(filename)

        if img_path is None:
            print(f"  [{class_name}] Image file '{filename}' not found on disk — skipping.")
            continue

        print(f"  [{class_name}] Using: {filename}")
        generate_gradcam_for_sample(
            model          = model,
            image_path     = img_path,
            class_idx      = class_idx,
            class_name     = class_name,
            exp_name       = exp_name,
            val_transforms = val_transforms,
            device         = DEVICE,
            save_dir       = config.GRADCAM_DIR,
        )

    print(f"\nAll Grad-CAM images saved to: {config.GRADCAM_DIR}")


if __name__ == "__main__":
    main()
