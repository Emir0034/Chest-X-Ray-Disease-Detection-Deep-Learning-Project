import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from transforms import get_train_transforms, get_val_transforms


# Convert "DiseaseA|DiseaseB" -> multi-hot float tensor of shape (14,)
def encode_labels(finding_labels: str, disease_labels: list) -> torch.Tensor:
    label_vec = torch.zeros(len(disease_labels), dtype=torch.float32)
    if not isinstance(finding_labels, str) or finding_labels.strip() == "No Finding":
        return label_vec
    for finding in finding_labels.split("|"):
        finding = finding.strip()
        if finding in disease_labels:
            label_vec[disease_labels.index(finding)] = 1.0
    return label_vec


# Extract patient ID from filename: "00000001_000.png" -> "00000001"
def extract_patient_id(image_index: str) -> str:
    return image_index.split("_")[0]


# Create or load patient-level train / val / test splits
# train=70%, val=15%, test=15% 
def create_patient_splits(
    csv_path: str,
    split_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:

    train_path = os.path.join(split_dir, "train_split.csv")
    val_path   = os.path.join(split_dir, "val_split.csv")
    test_path  = os.path.join(split_dir, "test_split.csv")

    #  Load existing splits to keep all experiments comparable if available, otherwise create new ones
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"[dataset] Loading existing splits from {split_dir}")
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    print(f"[dataset] Creating new patient-level splits from {csv_path}")
    df = pd.read_csv(csv_path)

    # Use Patient ID column if present, otherwise derive from filename
    if "Patient ID" in df.columns:
        df["_pid"] = df["Patient ID"].astype(str)
    else:
        df["_pid"] = df["Image Index"].apply(extract_patient_id)

    unique_patients = sorted(df["_pid"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)

    train_patients = set(unique_patients[:n_train])
    val_patients   = set(unique_patients[n_train : n_train + n_val])
    test_patients  = set(unique_patients[n_train + n_val :])

    train_df = df[df["_pid"].isin(train_patients)].drop(columns=["_pid"]).reset_index(drop=True)
    val_df   = df[df["_pid"].isin(val_patients)].drop(columns=["_pid"]).reset_index(drop=True)
    test_df  = df[df["_pid"].isin(test_patients)].drop(columns=["_pid"]).reset_index(drop=True)

    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(
        f"[dataset] Split complete — "
        f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} images"
    )
    return train_df, val_df, test_df


# Read a NIH split list file (one image filename per line) into a set
def _read_image_list(list_path: str) -> set:
    with open(list_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


# Create or load splits based on the official NIH train_val_list.txt / test_list.txt
# test_list.txt -> test set
# train_val_list.txt -> patient-level split into train (90%) / val (10%)
def create_official_nih_splits(
    csv_path: str,
    train_val_list: str,
    test_list: str,
    split_dir: str,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple:

    train_path = os.path.join(split_dir, "train_split.csv")
    val_path   = os.path.join(split_dir, "val_split.csv")
    test_path  = os.path.join(split_dir, "test_split.csv")

    # Load existing official splits if available
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"[dataset] Loading existing official NIH splits from {split_dir}")
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    print(f"[dataset] Creating new official NIH splits from {csv_path}")
    df = pd.read_csv(csv_path)
    df["_pid"] = df["Patient ID"].astype(str) if "Patient ID" in df.columns else df["Image Index"].apply(extract_patient_id)

    train_val_images = _read_image_list(train_val_list)
    test_images      = _read_image_list(test_list)

    train_val_df = df[df["Image Index"].isin(train_val_images)].reset_index(drop=True)
    test_df_full = df[df["Image Index"].isin(test_images)].reset_index(drop=True)

    # Patient-level 90/10 split of train_val into train/val
    unique_patients = sorted(train_val_df["_pid"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)

    n       = len(unique_patients)
    n_val   = int(val_ratio * n)
    val_patients   = set(unique_patients[:n_val])
    train_patients = set(unique_patients[n_val:])
    test_patients  = set(test_df_full["_pid"].unique())

    # Leakage checks - train/val/test patient sets must be pairwise disjoint
    assert train_patients.isdisjoint(val_patients), "Patient leakage between train and val splits"
    assert (train_patients | val_patients).isdisjoint(test_patients), "Patient leakage between train_val and test splits"

    train_df = train_val_df[train_val_df["_pid"].isin(train_patients)].drop(columns=["_pid"]).reset_index(drop=True)
    val_df   = train_val_df[train_val_df["_pid"].isin(val_patients)].drop(columns=["_pid"]).reset_index(drop=True)
    test_df  = test_df_full.drop(columns=["_pid"]).reset_index(drop=True)

    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(
        f"[dataset] Official NIH split complete — "
        f"train: {len(train_df)} images / {len(train_patients)} patients, "
        f"val: {len(val_df)} images / {len(val_patients)} patients, "
        f"test: {len(test_df)} images / {len(test_patients)} patients"
    )
    print("[dataset] Verified: no patient overlap between train/val/test")
    return train_df, val_df, test_df


# Dispatch to the configured split mode and return (train_df, val_df, test_df)
def load_splits(cfg) -> tuple:
    if getattr(cfg, "SPLIT_MODE", "custom") == "official_nih":
        return create_official_nih_splits(
            csv_path       = cfg.DATA_CSV,
            train_val_list = cfg.NIH_TRAIN_VAL_LIST,
            test_list      = cfg.NIH_TEST_LIST,
            split_dir      = cfg.OFFICIAL_SPLIT_DIR,
            val_ratio      = cfg.OFFICIAL_VAL_RATIO,
            seed           = cfg.SEED,
        )
    return create_patient_splits(
        csv_path  = cfg.DATA_CSV,
        split_dir = cfg.SPLIT_DIR,
        seed      = cfg.SEED,
    )


# Store all file paths in a dict for fast access O(1)
def _build_path_index(images_dir: str) -> dict:

    index = {}
    for dirpath, _, filenames in os.walk(images_dir):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                index[fname] = os.path.join(dirpath, fname)
    return index


# Dataset for NIH ChestX-ray14 multi-label classification
class ChestXRayDataset(Dataset):

    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df           = df.reset_index(drop=True)
        self.transform    = transform
        self._path_index  = _build_path_index(images_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row      = self.df.iloc[idx]
        filename = row["Image Index"]
        img_path = self._path_index.get(filename)
        if img_path is None:
            raise FileNotFoundError(
                f"Image '{filename}' not found under the images directory. "
                "Check that IMAGES_DIR in config.py points to the correct location."
            )
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = encode_labels(row["Finding Labels"], config.DISEASE_LABELS)
        return image, label


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str,
    use_clahe: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple:

    # build DataLoaders for train, validation and test splits
    train_dataset = ChestXRayDataset(
        train_df, images_dir,
        transform=get_train_transforms(use_clahe, config.IMAGE_SIZE),
    )
    val_dataset = ChestXRayDataset(
        val_df, images_dir,
        transform=get_val_transforms(use_clahe, config.IMAGE_SIZE),
    )
    test_dataset = ChestXRayDataset(
        test_df, images_dir,
        transform=get_val_transforms(use_clahe, config.IMAGE_SIZE),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
