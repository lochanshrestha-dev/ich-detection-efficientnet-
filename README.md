readme = """# ICH Detection — EfficientNet-B4

AI-based intracranial hemorrhage detection and subtype classification 
from non-contrast CT scans.

## Results
- RSNA internal test: Macro AUC 0.9835 (95% CI 0.9821–0.9847)
- CQ500 external validation: Macro AUC 0.7276 (95% CI 0.6729–0.7815)

## Pipeline
- Model: EfficientNet-B4 (timm), 6-class multi-label
- Training: RSNA 2019 (526,962 slices)
- External validation: CQ500 (473 studies)
- Explainability: Grad-CAM on last convolutional block

## Files
- train.py — training pipeline
- dataset.py — JPEG dataloader
- preprocess.py — patient-level split
- gradcam.py — Grad-CAM visualization
- evaluate.py — external validation
- convert_to_jpg.py — DICOM to JPEG conversion

## Author
Dr. Lochan Shrestha, PAHS, Nepal
"""

with open('/kaggle/working/README.md', 'w') as f:
    f.write(readme)
print("README created")
