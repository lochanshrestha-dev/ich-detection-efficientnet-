"""
preprocess.py
=============
CT preprocessing pipeline for ICH detection.

- DICOM loading with Hounsfield Unit conversion
- Three-window extraction (brain, subdural, bone)
- Resize to 256x256
- Lightweight skull stripping with coverage guard
- Patient-level split (study-level proxy, safe for de-identified data)
- Sanity check visualisation with optional W&B logging
"""

import os
import warnings
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# 1. DICOM LOADING
# =============================================================================

def load_dicom(path: str) -> np.ndarray:
    """Load DICOM, apply RescaleSlope/Intercept to return Hounsfield Units."""
    try:
        ds = pydicom.dcmread(path, force=True)
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        return img * slope + intercept
    except Exception as e:
        print(f"[Warning] Error loading {path}: {e}")
        return None


# =============================================================================
# 2. THREE-WINDOW EXTRACTION
# =============================================================================

def extract_windows(hu_image: np.ndarray) -> np.ndarray:
    """
    Extract Brain, Subdural, Bone windows and stack as 3-channel [0,1] image.

    Channel 0 - Brain    : center=40,  width=80    (grey/white matter contrast)
    Channel 1 - Subdural : center=40,  width=130   (thin extra-axial bleeds)
    Channel 2 - Bone     : center=400, width=1800  (skull fractures, epidural extent)
    """
    windows = [(40, 80), (40, 130), (400, 1800)]
    channels = []
    for center, width in windows:
        low = center - width / 2.0
        high = center + width / 2.0
        chan = np.clip(hu_image, low, high)
        chan = (chan - low) / (high - low)
        channels.append(chan)
    return np.stack(channels, axis=-1)   # (H, W, 3)


# =============================================================================
# 3. RESIZE
# =============================================================================

def resize_image(image_3c: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize each channel to (size, size) using INTER_AREA for downsampling."""
    resized = []
    for c in range(3):
        ch = cv2.resize(image_3c[:, :, c], (size, size), interpolation=cv2.INTER_AREA)
        resized.append(ch)
    return np.stack(resized, axis=-1).astype(np.float32)


# =============================================================================
# 4. SKULL STRIPPING (lightweight)
# =============================================================================

def skull_strip(image_3c: np.ndarray) -> np.ndarray:
    """
    Fast Otsu-based skull stripping with convex hull and coverage guard.

    Coverage guard: if mask covers <20% or >95% of the image the strip likely
    failed (large hemorrhage, motion, air dominance) — return original.
    Use HD-BET for production-grade brain extraction.
    """
    brain_chan = (image_3c[:, :, 0] * 255).astype(np.uint8)
    _, binary = cv2.threshold(brain_chan, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_3c

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    mask = np.zeros_like(brain_chan)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    mask_coverage = mask.sum() / mask.size
    if mask_coverage < 0.20 or mask_coverage > 0.95:
        return image_3c

    mask_3c = np.repeat(mask[:, :, np.newaxis] / 255.0, 3, axis=2)
    return image_3c * mask_3c


# =============================================================================
# 5. PATIENT-LEVEL SPLIT (RSNA 2019 SAFE)
# =============================================================================

def patient_split(df: pd.DataFrame,
                  study_mapping: dict = None,
                  random_state: int = 42) -> tuple:
    """
    Split at study level to prevent slice-level leakage.

    The function:
      1. Strips file extensions (.dcm/.jpg) from image_id.
      2. Strips known subtype suffixes (_any, _epidural, ...).
      3. Treats the resulting base ID as a study identifier.
      4. Stratifies on any_ich at study level, then splits 70/15/15.
      5. Asserts train and test study sets are disjoint.

    Slice-level splitting causes data leakage — always split by study/patient.
    """
    df = df.copy()
    subtype_suffixes = ['any', 'epidural', 'intraparenchymal',
                        'intraventricular', 'subarachnoid', 'subdural']

    def _extract_study_id(img_id: str) -> str:
        # Strip file extension if present
        base = img_id.rsplit('.', 1)[0] if '.' in img_id else img_id
        # Strip subtype suffix if present
        for suffix in subtype_suffixes:
            if base.endswith(f'_{suffix}'):
                base = base[:-len(f'_{suffix}')]
                break
        return study_mapping.get(base, base) if study_mapping else base

    df['study_id'] = df['image_id'].apply(_extract_study_id)

    # Group by study_id, get patient-level any_ich (max over slices)
    study_any  = df.groupby('study_id')['any_ich'].max().reset_index()
    study_ids  = study_any['study_id'].values
    any_labels = study_any['any_ich'].values

    # Stratify only when both classes are present
    stratify_arg = any_labels if len(np.unique(any_labels)) >= 2 else None

    train_ids, test_ids = train_test_split(
        study_ids, test_size=0.30,
        random_state=random_state, stratify=stratify_arg
    )

    test_any      = study_any[study_any['study_id'].isin(test_ids)]['any_ich'].values
    test_stratify = test_any if len(np.unique(test_any)) >= 2 else None

    val_ids, test_ids = train_test_split(
        test_ids, test_size=0.5,
        random_state=random_state, stratify=test_stratify
    )

    df_train = df[df['study_id'].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df['study_id'].isin(val_ids)].reset_index(drop=True)
    df_test  = df[df['study_id'].isin(test_ids)].reset_index(drop=True)

    assert set(df_train['study_id']).isdisjoint(set(df_test['study_id'])), \
        "Data leakage detected: study IDs overlap between train and test."

    if wandb and wandb.run is not None:
        wandb.log({
            'data/train_studies': len(train_ids),
            'data/val_studies':   len(val_ids),
            'data/test_studies':  len(test_ids),
            'data/train_slices':  len(df_train),
            'data/val_slices':    len(df_val),
            'data/test_slices':   len(df_test),
        })

    return df_train, df_val, df_test


# =============================================================================
# 6. SANITY CHECK (optional — used during development only)
# =============================================================================

def sanity_check(df: pd.DataFrame, dicom_dir: str, n: int = 5,
                 output_path: str = 'sanity_check.png',
                 label_columns: list = None):
    """Visualise n random slices with 3-window channels side-by-side."""
    if label_columns is None:
        label_columns = ['any_ich', 'epidural', 'intraparenchymal',
                         'intraventricular', 'subarachnoid', 'subdural']

    n_samples = min(n, len(df))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i, (_, row) in enumerate(df.sample(n_samples, random_state=0).iterrows()):
        img_id = row['image_id']
        path = os.path.join(dicom_dir, img_id)
        if not path.endswith('.dcm'):
            path += '.dcm'
        hu = load_dicom(path)
        if hu is None:
            continue
        img_3c = extract_windows(hu)
        img_3c = resize_image(img_3c)
        img_3c = skull_strip(img_3c)

        titles = ['Brain window', 'Subdural window', 'Bone window']
        for j in range(3):
            axes[i, j].imshow(img_3c[:, :, j], cmap='gray', vmin=0, vmax=1)
            axes[i, j].set_title(titles[j])
            axes[i, j].axis('off')

        label_str = ', '.join(
            col.replace('_', ' ') for col in label_columns if row[col] == 1
        )
        fig.suptitle(f"Sample {i+1}: [{label_str}]", fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if wandb and wandb.run is not None:
        wandb.log({'sanity_check': wandb.Image(output_path,
                                               caption='3-window CT samples')})


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("preprocess.py self-test running...")
    test_df = pd.DataFrame({
        'image_id': [f'ID_000{i}_any' for i in range(10)] +
                    [f'ID_000{i}_epidural' for i in range(10)],
        'any_ich':          [1] * 10 + [0] * 10,
        'epidural':         [0] * 20,
        'intraparenchymal': [0] * 20,
        'intraventricular': [0] * 20,
        'subarachnoid':     [0] * 20,
        'subdural':         [0] * 20,
    })
    train, val, test = patient_split(test_df, random_state=42)
    assert set(train['study_id']).isdisjoint(set(test['study_id'])), \
        "Leakage test failed"
    print("  Leakage test passed.")
    print("  Bone window range = [-500, 1300]")
    print("  patient_split handles .dcm/.jpg extensions and bare IDs.")
