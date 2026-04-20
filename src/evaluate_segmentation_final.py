from pathlib import Path
import math

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GT_DIR = PROJECT_ROOT / "data" / "processed_masks" / "final_v4"
PRED_ROOT = PROJECT_ROOT / "outputs" / "classical_baseline_final"

METHOD_DIRS = {
    "hsv": PRED_ROOT / "hsv_masks",
    "ycrcb": PRED_ROOT / "ycrcb_masks",
    "intersection": PRED_ROOT / "intersection_masks",
    "unet": PROJECT_ROOT / "outputs" / "unet_final" / "predictions",
}

SPLITS = ["val", "test"]


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return (mask > 127).astype(np.uint8)


def compute_confusion(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((gt == 1) & (pred == 1)))
    tn = int(np.sum((gt == 0) & (pred == 0)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    return tp, tn, fp, fn


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp, tn, fp, fn = compute_confusion(gt, pred)

    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div((tp * tn) - (fp * fn), mcc_den)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    for method_name, method_dir in METHOD_DIRS.items():
        print(f"\n=== {method_name.upper()} ===")

        for split in SPLITS:
            split_dir = method_dir / split
            pred_paths = sorted(split_dir.glob("*.png"))

            if not pred_paths:
                print(f"{split}: no prediction files found")
                continue

            all_metrics = {
                "iou": [],
                "dice": [],
                "precision": [],
                "recall": [],
                "specificity": [],
                "balanced_accuracy": [],
                "mcc": [],
            }

            missing_gt = 0

            for pred_path in pred_paths:
                image_id = pred_path.stem
                gt_path = GT_DIR / f"{image_id}.png"

                if not gt_path.exists():
                    missing_gt += 1
                    continue

                gt = load_binary_mask(gt_path)
                pred = load_binary_mask(pred_path)

                if gt.shape != pred.shape:
                    pred = cv2.resize(
                        pred,
                        (gt.shape[1], gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    pred = (pred > 0).astype(np.uint8)

                metrics = compute_metrics(gt, pred)

                for key, value in metrics.items():
                    all_metrics[key].append(value)

            print(f"\n{split}:")
            print(f"  evaluated images: {len(all_metrics['iou'])}")
            print(f"  missing gt masks: {missing_gt}")

            for metric_name, values in all_metrics.items():
                mean_val, std_val = summarize(values)
                print(f"  {metric_name}: mean={mean_val:.4f}, std={std_val:.4f}")


if __name__ == "__main__":
    main()