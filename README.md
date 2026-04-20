# Conservative Facial Skin Segmentation Under Domain Shift

**IMGS-621 Computer Vision Project**  
Author: Anjana Parasuram  
Rochester Institute of Technology

---

## Overview

This project investigates robust **binary facial skin segmentation** under changing image conditions. The goal is to isolate stable exposed skin regions from portrait images while handling ambiguity caused by makeup, hair, shadows, facial features, and boundary noise.

Rather than relying only on benchmark performance, this project also frames the task as a **domain shift problem**: models trained on curated celebrity imagery may struggle on real phone selfies with different lighting, cameras, backgrounds, and poses.

### Input / Output

- **Input:** RGB portrait image  
- **Output:** Binary segmentation mask  
- **Classes:** Skin vs Non-skin

---

## Motivation

Standard semantic face masks often include regions that are undesirable for downstream skin analysis:

- eyebrows  
- eyelashes  
- eyes  
- lips  
- mouth interior  
- nose contours  
- hair overlap  
- ambiguous facial boundaries

For applications like skin tone estimation or cosmetic shade matching, these areas can introduce instability.

This project therefore redefines the segmentation target toward:

> **Stable exposed planar skin regions with lower ambiguity**

Examples:

- forehead  
- cheeks  
- jawline  
- neck (when available)

---

## Conservative Label Engineering

### Base Semantic Logic

```text
core skin =
skin
- brows
- eyes
- lips
- mouth
- nose
- hair
```

### Morphological Refinement

Two morphological operations were used:

1. **Dilation of excluded regions**  
   Expands unstable regions to create safety margins.

2. **Erosion of retained skin**  
   Restricts the final mask to high-confidence interior pixels.

### Final Chosen Target (V4)

```python
medium_kernel_size = 7
medium_iterations = 1
eye_kernel_size = 29
eye_iterations = 1
final_erosion = 3
```

---

## Why Remove the Nose?

The nose introduces several challenges:

- strong highlights and shadows  
- high curvature geometry  
- unstable appearance across pose changes  
- ambiguous cheek boundary transitions

Removing the nose improves mask consistency for downstream color analysis.

---

## Dataset

**CelebAMask-HQ** (30,000 images)

Official CelebA split preserved:

| Split | Images |
|------|------:|
| Train | 24,183 |
| Val | 2,993 |
| Test | 2,824 |

---

## Methods Compared

### Classical Baselines

- HSV thresholding  
- YCrCb thresholding  
- Intersection of HSV + YCrCb

### Learned Model

- **U-Net** with ResNet18 encoder

---

## Evaluation Metrics

Pixelwise TP / TN / FP / FN were accumulated across images.

**IoU = TP / (TP + FP + FN)**  
Measures how much the predicted skin region overlaps the conservative ground-truth mask relative to their combined area.

**Dice = 2TP / (2TP + FP + FN)**  
Measures overall agreement between predicted and true skin regions, with emphasis on correctly overlapping skin pixels.

**Precision = TP / (TP + FP)**  
Measures how often pixels predicted as skin were actually true skin, so low precision indicates over-segmentation into hair, lips, shadows, or background.

**Recall = TP / (TP + FN)**  
Measures how much of the true skin region was successfully recovered, so low recall means valid skin areas were missed.

**Specificity = TN / (TN + FP)**  
Measures how well the model correctly rejects non-skin pixels such as hair, clothing, background, or excluded facial parts.

**Balanced Accuracy = (Recall + Specificity) / 2**  
Measures average performance across both skin and non-skin classes, helping account for class imbalance where background pixels dominate.

**MCC = ((TP·TN) - (FP·FN)) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))**  
Measures overall segmentation quality using all four confusion-matrix terms and is especially useful when skin and background pixels are highly imbalanced.

---

## Results

### Test Set Performance

| Method | IoU | Dice | Precision | Recall | Specificity | Balanced Acc | MCC |
|------|----:|----:|----:|----:|----:|----:|----:|
| HSV | 0.3663 | 0.5229 | 0.3836 | 0.9156 | 0.5948 | 0.7552 | 0.4230 |
| YCrCb | 0.3154 | 0.4663 | 0.3210 | 0.9542 | 0.4402 | 0.6972 | 0.3369 |
| Intersection | 0.3660 | 0.5217 | 0.3860 | 0.8997 | 0.6069 | 0.7533 | 0.4200 |
| **U-Net** | **0.9065** | **0.9501** | **0.9499** | **0.9518** | **0.9876** | **0.9697** | **0.9385** |

---

## Key Interpretation

### Major Finding

Learned spatial segmentation strongly outperformed classical color-threshold heuristics.

### Improvement over Best Baseline (HSV)

- **IoU:** 0.3663 → 0.9065  
- **Dice:** 0.5229 → 0.9501  
- **MCC:** 0.4230 → 0.9385  

### Why U-Net Wins

Unlike pixelwise thresholding, U-Net learns:

- facial geometry  
- semantic boundaries  
- contextual continuity  
- exclusion of unstable regions  
- robust conservative contours

---

## Generalization Check

Validation and test metrics were nearly identical:

| Metric | Val | Test |
|------|----:|----:|
| IoU | 0.9078 | 0.9065 |
| Dice | 0.9511 | 0.9501 |

This suggests:

- minimal overfitting  
- stable training  
- strong held-out generalization

---

## Limitations

- props / occlusions (hands, microphones, glasses)  
- heavy makeup / eyelashes  
- facial hair not explicitly removed  
- real-world lighting shift remains challenging

---

## Future Work

- evaluate on real selfie images  
- compare DeepLabV3+, Attention U-Net, SAM variants  
- estimate skin tone using CIE Lab after segmentation  
- improve robustness under illumination shift

---
## Repository Structure

```text
.
├── data/              # Contains dataset assets (images, masks, metadata, splits), download from Google Drive.
├── outputs/           # Generated predictions, overlays, evaluation outputs, and trained checkpoints, download from Google Drive.
├── src/               # Source code for preprocessing, training, inference, and evaluation.
├── requirements.txt   # Python dependencies needed to reproduce the project environment.
└── README.md          # Project report, methodology, results, and usage instructions.

data/
├── metadata/            # Official CelebA / CelebA-HQ mapping and split metadata
├── raw_images/          # Original RGB portrait images
│   ├── celeba/          # CelebAMask-HQ benchmark images
│   └── shifted_selfies/ # Optional harder real-world test images
├── raw_masks/           # Original semantic benchmark masks
├── processed_masks/     # Custom engineered conservative masks
│   ├── pilot/           # Pilot mask experiments (V1–V4)
│   └── final_v4/        # Final training masks used for U-Net
└── splits/              # Train / val / test ID lists

outputs/
├── classical_baseline_final/   # HSV / YCrCb / Intersection predictions
├── pilot_overlays/             # Visual overlays for pilot mask selection
└── unet_final/
    ├── models/                 # Trained U-Net checkpoints
    ├── predictions/            # Saved predicted masks (val/test)
    └── overlays/               # Visualization overlays of predictions
    
src/
├── build_conservative_mask_pilot.py   # Creates pilot mask candidates
├── build_final_v4_dataset.py          # Generates final conservative masks
├── build_official_splits.py           # Builds official train/val/test splits
├── run_classical_baseline_final.py    # Runs HSV / YCrCb baselines
├── train_unet_final.py                # Trains final U-Net model
├── predict_unet_final.py              # Runs inference using trained model
├── evaluate_segmentation_final.py     # Computes IoU, Dice, MCC
└── make_unet_overlays.py              # Creates qualitative visualization overlays

Full data and outputs folder available here:
https://drive.google.com/drive/folders/1n1RLtGvbZ7ux8xICtuUjXizDJMyAxz8B?usp=sharing

```markdown
## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Conclusion

Careful label engineering combined with supervised segmentation substantially outperformed classical threshold-based skin detection.

The resulting U-Net achieved excellent overlap performance and strong held-out generalization, demonstrating that robust facial skin isolation is feasible and promising for downstream appearance analysis tasks.
