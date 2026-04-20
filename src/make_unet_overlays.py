from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images" / "celeba"
PRED_DIR = PROJECT_ROOT / "outputs" / "unet_final" / "predictions"
OUT_DIR = PROJECT_ROOT / "outputs" / "unet_final" / "overlays"

SPLITS = ["val", "test"]


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)


def main() -> None:
    for split in SPLITS:
        pred_split_dir = PRED_DIR / split
        out_split_dir = OUT_DIR / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        pred_paths = sorted(pred_split_dir.glob("*.png"))

        for idx, pred_path in enumerate(pred_paths, start=1):
            image_id = pred_path.stem
            image_path = IMAGE_DIR / f"{image_id}.jpg"

            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Skipping {image_id}")
                continue

            mask_bin = (mask > 127).astype(np.uint8)
            overlay = make_overlay(image, mask_bin)

            cv2.imwrite(str(out_split_dir / f"{image_id}.png"), overlay)

            if idx % 500 == 0:
                print(f"{split}: processed {idx}/{len(pred_paths)} overlays")

    print("Done saving overlays.")


if __name__ == "__main__":
    main()