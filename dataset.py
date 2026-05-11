"""
dataset.py
==========
ICH Dataset loading pre-converted JPEG files.
Drop-in replacement for DICOM loader. ~10x faster I/O.

Use after running convert_to_jpg.py and uploading the JPEG datasets to Kaggle.
Supports multiple JPEG directories (search across mounted Kaggle datasets).
"""

import os
import platform
import warnings
from typing import Union, List

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# DEVICE & W&B UTILS
# =============================================================================

def get_device() -> torch.device:
    """Return best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def init_wandb(project: str = 'ich-detection', config: dict = None):
    """Initialise W&B with preprocessing/training config."""
    default_config = {
        'preprocessing': {
            'loader':        'JPEG (q=95)',
            'resize':        256,
            'normalization': 'ImageNet',
        },
        'augmentation': ['HorizontalFlip', 'Rotate±10', 'BrightnessContrast'],
    }
    if config:
        default_config.update(config)
    if wandb:
        wandb.init(project=project, config=default_config)


# =============================================================================
# DATASET
# =============================================================================

class ICHDataset(Dataset):
    """
    PyTorch Dataset for multi-label ICH classification using pre-converted JPEGs.

    Each JPEG is 256x256x3 uint8 with channels:
        ch0 = Brain window
        ch1 = Subdural window
        ch2 = Bone window

    DataFrame must contain columns:
        image_id, any_ich, epidural, intraparenchymal,
        intraventricular, subarachnoid, subdural

    Args:
        df       : slice-level DataFrame
        jpg_dirs : single directory or list of directories of .jpg files
        transform: apply augmentation (train mode only)
        mode     : 'train', 'val', or 'test'
    """

    LABEL_COLS = ['any_ich', 'epidural', 'intraparenchymal',
                  'intraventricular', 'subarachnoid', 'subdural']

    def __init__(self,
                 df: pd.DataFrame,
                 jpg_dirs: Union[str, List[str]],
                 transform: bool = True,
                 mode: str = 'train'):
        self.df       = df.reset_index(drop=True)
        self.jpg_dirs = jpg_dirs if isinstance(jpg_dirs, list) else [jpg_dirs]
        self.mode     = mode

        if transform and mode == 'train':
            self.transform_fn = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform_fn = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row    = self.df.iloc[idx]
        img_id = str(row['image_id'])

        # Strip any extension and add .jpg
        base = img_id.rsplit('.', 1)[0] if '.' in img_id else img_id

        # Search across all jpg directories
        img = None
        for d in self.jpg_dirs:
            jpg_path = os.path.join(d, base + '.jpg')
            if os.path.exists(jpg_path):
                img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
                break

        if img is None:
            warnings.warn(f"Image {img_id} not found in jpg_dirs. Using zero tensor.")
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # BGR -> RGB to match channel expectations (brain=R, subdural=G, bone=B)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed   = self.transform_fn(image=img)
        image_tensor  = transformed['image']

        labels = torch.tensor(
            row[self.LABEL_COLS].values.astype(np.float32))

        return {
            'image':      image_tensor,
            'label':      labels,
            'patient_id': str(row.get('study_id', img_id)),
            'slice_id':   img_id,
        }


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    weight[i] = total / (num_positives[i] * num_classes)
    """
    label_cols  = ICHDataset.LABEL_COLS
    total       = len(df)
    pos_counts  = df[label_cols].sum(axis=0).values
    num_classes = len(label_cols)
    weights     = total / (pos_counts * num_classes + 1e-6)

    print("Class distribution and weights:")
    for col, cnt, w in zip(label_cols, pos_counts, weights):
        print(f"  {col:25s}: {int(cnt):7,d} ({cnt/total*100:5.1f}%)  weight={w:.4f}")

    if wandb and wandb.run is not None:
        wandb.log({col: int(cnt) for col, cnt in zip(label_cols, pos_counts)})
        wandb.log({f'class_weight/{col}': float(w)
                   for col, w in zip(label_cols, weights)})

    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def get_dataloaders(df_train: pd.DataFrame,
                    df_val:   pd.DataFrame,
                    df_test:  pd.DataFrame,
                    jpg_dirs: Union[str, List[str]],
                    batch_size: int = 32,
                    num_workers: int = None) -> dict:
    """
    Build train/val/test DataLoaders from pre-converted JPEG directories.

    num_workers (auto when None):
        macOS  -> 0  (MPS + multiprocessing incompatible)
        other  -> 4  (Kaggle T4; JPEG loading is light on I/O)

    pin_memory: True only when CUDA is available.
    """
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Darwin' else 4
    pin_memory = torch.cuda.is_available()

    datasets = {
        'train': ICHDataset(df_train, jpg_dirs, transform=True,  mode='train'),
        'val':   ICHDataset(df_val,   jpg_dirs, transform=False, mode='val'),
        'test':  ICHDataset(df_test,  jpg_dirs, transform=False, mode='test'),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == 'train'),
        )
    return loaders


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import tempfile

    print("dataset.py self-test running...")
    print(f"  device      : {get_device()}")
    print(f"  num_workers : {0 if platform.system() == 'Darwin' else 4}")
    print(f"  pin_memory  : {torch.cuda.is_available()}")

    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        # Write a synthetic JPEG only to dir2 to test multi-dir search
        fake = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(
            os.path.join(dir2, 'ID_test0001.jpg'),
            fake,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        df = pd.DataFrame({
            'image_id':         ['ID_test0001'],   # bare ID — no extension
            'any_ich':          [1],
            'epidural':         [0],
            'intraparenchymal': [1],
            'intraventricular': [0],
            'subarachnoid':     [0],
            'subdural':         [0],
        })

        ds   = ICHDataset(df, [dir1, dir2], transform=False, mode='val')
        item = ds[0]

        assert item['image'].shape == (3, 256, 256), \
            f"Wrong image shape: {item['image'].shape}"
        assert item['label'].shape == (6,), \
            f"Wrong label shape: {item['label'].shape}"

        print(f"  image shape : {item['image'].shape}  OK")
        print(f"  label       : {item['label'].tolist()}  OK")

        w = compute_class_weights(df)
        print(f"  class weights (first 3): {w[:3].tolist()}  OK")

    print("  All dataset checks passed.")
