from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images" / "celeba"
MASK_DIR = PROJECT_ROOT / "data" / "processed_masks" / "final_v4"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

OUT_DIR = PROJECT_ROOT / "outputs" / "classical_baseline_final"
OUT_HSV = OUT_DIR / "hsv_masks"
OUT_YCRCB = OUT_DIR / "ycrcb_masks"
OUT_INTERSECTION = OUT_DIR / "intersection_masks"

VAL_IDS_PATH = SPLIT_DIR / "val_ids.txt"
TEST_IDS_PATH = SPLIT_DIR / "test_ids.txt"

def load_ids(txt_path: Path) -> list[str]:
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    return mask


def skin_mask_hsv(image_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    return clean_mask(mask)


def skin_mask_ycrcb(image_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)
    return clean_mask(mask)


def skin_mask_intersection(image_bgr: np.ndarray) -> np.ndarray:
    hsv_mask = skin_mask_hsv(image_bgr)
    ycrcb_mask = skin_mask_ycrcb(image_bgr)

    intersection = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    return clean_mask(intersection)


def main() -> None:
    for folder in [OUT_HSV, OUT_YCRCB, OUT_INTERSECTION]:
        folder.mkdir(parents=True, exist_ok=True)

    val_ids = load_ids(VAL_IDS_PATH)
    test_ids = load_ids(TEST_IDS_PATH)

    eval_ids = {
        "val": val_ids,
        "test": test_ids,
    }

    for split_name, image_ids in eval_ids.items():
        print(f"{split_name}: {len(image_ids)} images")

        split_hsv_dir = OUT_HSV / split_name
        split_ycrcb_dir = OUT_YCRCB / split_name
        split_intersection_dir = OUT_INTERSECTION / split_name

        for folder in [split_hsv_dir, split_ycrcb_dir, split_intersection_dir]:
            folder.mkdir(parents=True, exist_ok=True)

        for idx, image_id in enumerate(image_ids, start=1):
            image_path = IMAGE_DIR / f"{image_id}.jpg"
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping missing image: {image_path.name}")
                continue

            hsv_mask = skin_mask_hsv(image)
            ycrcb_mask = skin_mask_ycrcb(image)
            intersection_mask = skin_mask_intersection(image)

            cv2.imwrite(str(split_hsv_dir / f"{image_id}.png"), hsv_mask)
            cv2.imwrite(str(split_ycrcb_dir / f"{image_id}.png"), ycrcb_mask)
            cv2.imwrite(str(split_intersection_dir / f"{image_id}.png"), intersection_mask)

            if idx % 500 == 0:
                print(f"  {split_name}: processed {idx}/{len(image_ids)}")


if __name__ == "__main__":
    main()