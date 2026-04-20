from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images" / "celeba"
MASK_ROOT = PROJECT_ROOT / "data" / "raw_masks" / "celeba_original"
OUT_DIR = PROJECT_ROOT / "data" / "processed_masks" / "final_v4"


MEDIUM_DILATE_PARTS = [
    "l_brow",
    "r_brow",
    "l_lip",
    "u_lip",
    "mouth",
]

EYE_PARTS = [
    "l_eye",
    "r_eye",
]

DIRECT_REMOVE_PARTS = [
    "nose",
    "hair",
]


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return (mask > 0).astype(np.uint8)


def to_mask_id(image_id: str) -> str:
    return image_id.zfill(5)


def find_part_mask_path(image_id: str, part_name: str) -> Path | None:
    mask_id = to_mask_id(image_id)

    for folder in sorted(MASK_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        candidate = folder / f"{mask_id}_{part_name}.png"
        if candidate.exists():
            return candidate

    return None


def dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return (dilated > 0).astype(np.uint8)


def build_union_mask(image_id: str, part_names: list[str]) -> np.ndarray | None:
    union_mask = None

    for part_name in part_names:
        part_path = find_part_mask_path(image_id, part_name)
        if part_path is None:
            continue

        part_mask = load_binary_mask(part_path)

        if union_mask is None:
            union_mask = part_mask
        else:
            union_mask = np.maximum(union_mask, part_mask)

    return union_mask


def build_semantic_core_mask(
    image_id: str,
    medium_kernel_size: int = 1,
    medium_iterations: int = 0,
    eye_kernel_size: int = 1,
    eye_iterations: int = 0,
) -> np.ndarray:
    skin_path = find_part_mask_path(image_id, "skin")
    if skin_path is None:
        raise FileNotFoundError(f"No skin mask found for image id: {image_id}")

    skin_mask = load_binary_mask(skin_path)

    remove_mask = np.zeros_like(skin_mask)

    medium_mask = build_union_mask(image_id, MEDIUM_DILATE_PARTS)
    if medium_mask is not None:
        if medium_iterations > 0:
            medium_mask = dilate_mask(
                medium_mask,
                kernel_size=medium_kernel_size,
                iterations=medium_iterations,
            )
        remove_mask = np.maximum(remove_mask, medium_mask)

    eye_mask = build_union_mask(image_id, EYE_PARTS)
    if eye_mask is not None:
        if eye_iterations > 0:
            eye_mask = dilate_mask(
                eye_mask,
                kernel_size=eye_kernel_size,
                iterations=eye_iterations,
            )
        remove_mask = np.maximum(remove_mask, eye_mask)

    direct_mask = build_union_mask(image_id, DIRECT_REMOVE_PARTS)
    if direct_mask is not None:
        remove_mask = np.maximum(remove_mask, direct_mask)

    core_mask = np.logical_and(skin_mask == 1, remove_mask == 0).astype(np.uint8)
    return core_mask


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    return (eroded > 0).astype(np.uint8)


def build_final_v4_mask(image_id: str) -> np.ndarray:
    mask_core = build_semantic_core_mask(
        image_id,
        medium_kernel_size=7,
        medium_iterations=1,
        eye_kernel_size=29,
        eye_iterations=1,
    )
    return erode_mask(mask_core, iterations=3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    print(f"Found {len(image_paths)} images.")

    saved_count = 0
    skipped_count = 0

    for idx, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem

        try:
            final_mask = build_final_v4_mask(image_id)
        except FileNotFoundError as e:
            print(f"Skipping {image_id}: {e}")
            skipped_count += 1
            continue

        save_path = OUT_DIR / f"{image_id}.png"
        ok = cv2.imwrite(str(save_path), final_mask * 255)

        if not ok:
            print(f"Failed to save mask for {image_id}")
            skipped_count += 1
            continue

        saved_count += 1

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)} images...")

    print(f"Done. Saved: {saved_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()