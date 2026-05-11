"""
convert_to_jpg.py
=================
One-time conversion of RSNA 2019 DICOM CT slices to 3-channel JPEG files.

Each JPEG encodes three CT windows stacked as RGB channels:
  R = Brain window    (center=40,  width=80)
  G = Subdural window (center=40,  width=130)
  B = Bone window     (center=400, width=1800)

JPEG quality=95 gives ~8-12 KB per slice vs ~50-70 KB for PNG.
674,000 slices × 10 KB = ~6.5 GB — well under Kaggle's 20GB dataset limit.

Run ONCE in a Kaggle CPU notebook. Save output folder as a new Kaggle dataset.
After conversion, training loads JPEGs in ~0.03s vs ~0.35s for DICOM — 10x faster.

KAGGLE CPU NOTEBOOK USAGE:
    !pip install pydicom opencv-python-headless tqdm -q
    !python convert_to_jpg.py \
        --dicom_dir /kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/ \
        --output_dir /kaggle/working/rsna_jpg/ \
        --num_workers 4

Estimated conversion time: 3-4 hours on Kaggle CPU (4 workers)
Estimated output size:     ~6-8 GB (JPEG q=95, 256x256)

OUTPUT STRUCTURE:
    rsna_jpg/
        ID_000039fa4.jpg
        ID_000012345.jpg
        ...
        failed.txt    <- slices that failed conversion (inspect if non-empty)
"""

import os
import argparse
import numpy as np
import pydicom
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


# ─── CT WINDOW SETTINGS ───────────────────────────────────────────────────────

WINDOWS = [
    (40,   80),   # Brain    — channel 0 (R)
    (40,  130),   # Subdural — channel 1 (G)
    (400, 1800),  # Bone     — channel 2 (B) — range [-500, 1300] HU
]

JPEG_QUALITY = 95   # 95 = near-lossless; artefacts imperceptible to model
                    # Lower to 90 if you need to save more space (~4 GB total)


# ─── WINDOWING ────────────────────────────────────────────────────────────────

def hu_to_window(hu_image: np.ndarray, center: int, width: int) -> np.ndarray:
    """Clip HU values to window range and normalise to uint8 [0, 255]."""
    low  = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(hu_image, low, high)
    normalised = (windowed - low) / (high - low)
    return (normalised * 255).astype(np.uint8)


# ─── DICOM → ARRAY ────────────────────────────────────────────────────────────

def dicom_to_array(dcm_path: str, size: int = 256) -> np.ndarray:
    """
    Load a DICOM, apply HU conversion, extract 3 windows, resize.

    Returns:
        np.ndarray (size, size, 3) uint8
    Raises:
        Exception on load or pixel array failure.
    """
    ds  = pydicom.dcmread(dcm_path, force=True)
    img = ds.pixel_array.astype(np.float32)

    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    hu = img * slope + intercept

    channels = []
    for center, width in WINDOWS:
        ch = hu_to_window(hu, center, width)
        ch = cv2.resize(ch, (size, size), interpolation=cv2.INTER_AREA)
        channels.append(ch)

    return np.stack(channels, axis=-1)   # (H, W, 3) uint8


# ─── WORKER (runs in subprocess) ──────────────────────────────────────────────

def convert_one(task: tuple) -> tuple:
    """
    Convert one DICOM to JPEG.

    Args:
        task: (dcm_path, jpg_path, size)

    Returns:
        (slice_id, 'ok' | 'skipped' | error_traceback)
    """
    dcm_path, jpg_path, size = task
    slice_id = os.path.splitext(os.path.basename(dcm_path))[0]
    try:
        if os.path.exists(jpg_path):
            return (slice_id, 'skipped')   # resume-friendly — skip already done

        arr = dicom_to_array(dcm_path, size=size)

        # OpenCV imwrite with JPEG quality parameter
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        success = cv2.imwrite(jpg_path, arr, encode_params)
        if not success:
            raise RuntimeError(f"cv2.imwrite failed for {jpg_path}")

        return (slice_id, 'ok')

    except Exception:
        return (slice_id, traceback.format_exc())


# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert RSNA DICOM CT slices to JPEG (3-channel windowed)"
    )
    parser.add_argument('--dicom_dir',   type=str, required=True,
                        help='Directory containing .dcm files')
    parser.add_argument('--output_dir',  type=str, default='/kaggle/working/rsna_jpg',
                        help='Output directory for .jpg files')
    parser.add_argument('--size',        type=int, default=256,
                        help='Output image size in pixels (square). Default: 256')
    parser.add_argument('--quality',     type=int, default=JPEG_QUALITY,
                        help=f'JPEG quality 0-100. Default: {JPEG_QUALITY}')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Parallel worker processes. Default: 4')
    parser.add_argument('--limit',       type=int, default=None,
                        help='Convert only first N files (for testing). Omit for full run.')
    return parser.parse_args()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    failed_log = os.path.join(args.output_dir, 'failed.txt')

    # ── Discover DICOM files ──
    all_dcm = sorted([
        f for f in os.listdir(args.dicom_dir)
        if f.lower().endswith('.dcm')
    ])

    if not all_dcm:
        # Try one level of subdirectories (some Kaggle dataset versions)
        print("No .dcm files at top level — scanning subdirectories...")
        for subdir in sorted(os.listdir(args.dicom_dir)):
            sub_path = os.path.join(args.dicom_dir, subdir)
            if os.path.isdir(sub_path):
                found = [
                    os.path.join(subdir, f)
                    for f in os.listdir(sub_path)
                    if f.lower().endswith('.dcm')
                ]
                all_dcm.extend(found)
        if all_dcm:
            print(f"Found {len(all_dcm):,} .dcm files in subdirectories.")

    if not all_dcm:
        print(f"ERROR: No .dcm files found in {args.dicom_dir}")
        return

    if args.limit:
        all_dcm = all_dcm[:args.limit]
        print(f"Limiting to first {args.limit} files (test mode).")

    total = len(all_dcm)
    est_gb = total * 10 / 1024 / 1024   # ~10 KB per JPEG at q=95
    est_min = total / (args.num_workers * 8) / 60

    print(f"\n{'─'*50}")
    print(f"  DICOM source : {args.dicom_dir}")
    print(f"  JPEG output  : {args.output_dir}")
    print(f"  Total slices : {total:,}")
    print(f"  Image size   : {args.size}×{args.size} px")
    print(f"  JPEG quality : {args.quality}")
    print(f"  Workers      : {args.num_workers}")
    print(f"  Est. size    : ~{est_gb:.1f} GB")
    print(f"  Est. time    : ~{est_min:.0f} minutes")
    print(f"{'─'*50}\n")

    # ── Build task list ──
    tasks = []
    for dcm_name in all_dcm:
        dcm_path  = os.path.join(args.dicom_dir, dcm_name)
        base_name = os.path.splitext(os.path.basename(dcm_name))[0]
        jpg_path  = os.path.join(args.output_dir, base_name + '.jpg')
        tasks.append((dcm_path, jpg_path, args.size))

    # ── Convert in parallel ──
    ok_count = skipped_count = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(convert_one, t): t for t in tasks}
        with tqdm(total=total, desc="Converting DICOMs", unit="slice") as pbar:
            for future in as_completed(futures):
                slice_id, result = future.result()
                if result == 'ok':
                    ok_count += 1
                elif result == 'skipped':
                    skipped_count += 1
                else:
                    failed.append((slice_id, result))
                pbar.update(1)
                pbar.set_postfix({
                    'ok':   ok_count,
                    'skip': skipped_count,
                    'fail': len(failed),
                })

    # ── Write failure log ──
    if failed:
        with open(failed_log, 'w') as f:
            for sid, err in failed:
                f.write(f"{sid}\n{err}\n{'─'*40}\n")
        print(f"\n  {len(failed):,} failures logged → {failed_log}")
        print("  Inspect failed.txt — common causes: corrupt DICOM, missing pixel data")
    else:
        print("\n  No failures.")

    # ── Final summary ──
    actual_gb = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith('.jpg')
    ) / 1e9

    print(f"\n{'─'*50}")
    print(f"  Converted : {ok_count:,}")
    print(f"  Skipped   : {skipped_count:,}  (already existed)")
    print(f"  Failed    : {len(failed):,}")
    print(f"  Actual size: {actual_gb:.2f} GB")
    print(f"  Output    : {args.output_dir}")
    print(f"{'─'*50}")
    print(f"""
NEXT STEPS:
  1. In this notebook's output panel → click 'New Dataset'
  2. Name it: rsna-ich-jpg-256
  3. In your GPU training notebook, add this dataset and set:
       --dicom_dir /kaggle/input/rsna-ich-jpg-256/rsna_jpg/
  4. Use dataset_jpg.py (rename to dataset.py) for 10x faster loading
  5. Run 50 epochs comfortably in one 12-hour session
""")


if __name__ == '__main__':
    main()
