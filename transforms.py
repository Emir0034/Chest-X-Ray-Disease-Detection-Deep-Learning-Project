import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class CLAHETransform:


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

    pipeline = []

    if use_clahe:
        pipeline.append(CLAHETransform())

    pipeline += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    return transforms.Compose(pipeline)
