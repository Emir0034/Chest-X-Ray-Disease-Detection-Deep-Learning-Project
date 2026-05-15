import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# ImageNet statistics used because we start from pretrained DenseNet121 weights.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a PIL image.

    Chest X-rays are effectively grayscale, so CLAHE is applied to the
    luminance channel only and the result is broadcast back to 3 channels.
    This avoids color-channel artefacts while improving local contrast.

    Args:
        clip_limit:     Threshold for contrast limiting (default 2.0).
        tile_grid_size: Size of the grid for histogram equalization (default 8×8).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)                                      # uint8 (H, W, 3)
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)           # (H, W)
        clahe  = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)                                 # (H, W)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)            # (H, W, 3)
        return Image.fromarray(rgb)


def get_train_transforms(use_clahe: bool = False, image_size: int = 224) -> transforms.Compose:
    """Return the augmentation pipeline for training.

    Medically safe augmentations only:
      - Small random resized crop (scale 0.8–1.0)
      - Rotation up to ±10° (no flips — flipping chest X-rays is anatomically wrong)
      - Subtle brightness / contrast jitter
    CLAHE is inserted at the front when use_clahe=True.
    """
    pipeline = []

    if use_clahe:
        pipeline.append(CLAHETransform())

    pipeline += [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    return transforms.Compose(pipeline)


def get_val_transforms(use_clahe: bool = False, image_size: int = 224) -> transforms.Compose:
    """Return the deterministic pipeline for validation and test.

    No random augmentations — only resize, optional CLAHE, and normalisation.
    """
    pipeline = []

    if use_clahe:
        pipeline.append(CLAHETransform())

    pipeline += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    return transforms.Compose(pipeline)
