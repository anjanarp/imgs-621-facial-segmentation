from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images" / "celeba"
MASK_ROOT = PROJECT_ROOT / "data" / "raw_masks" / "celeba_original"

OUT_V1 = PROJECT_ROOT / "data" / "processed_masks" / "pilot" / "v1_semantic_only"
OUT_V2 = PROJECT_ROOT / "data" / "processed_masks" / "pilot" / "v2_semantic_erode1"
OUT_V3 = PROJECT_ROOT / "data" / "processed_masks" / "pilot" / "v3_semantic_erode2"
OUT_V4 = PROJECT_ROOT / "data" / "processed_masks" / "pilot" / "v4_ultra_conservative"

OVERLAY_V1 = PROJECT_ROOT / "outputs" / "pilot_overlays" / "v1_semantic_only"
OVERLAY_V2 = PROJECT_ROOT / "outputs" / "pilot_overlays" / "v2_semantic_erode1"
OVERLAY_V3 = PROJECT_ROOT / "outputs" / "pilot_overlays" / "v3_semantic_erode2"
OVERLAY_V4 = PROJECT_ROOT / "outputs" / "pilot_overlays" / "v4_ultra_conservative"


PILOT_IMAGE_IDS = [
    "0",
    "10",
    "50",
    "100",
    "200",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "900",
    "1000",
    "2000",
    "3000",
    "4000",
    "5000",
    "6000",
    "7000",
    "8000",
]

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


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image_bgr.copy()
    overlay[mask_resized > 0] = (0, 255, 0)

    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

def main() -> None:
    for image_id in PILOT_IMAGE_IDS:
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping missing image: {image_path.name}")
            continue

        try:
            mask_v1 = build_semantic_core_mask(image_id)

            mask_v2_core = build_semantic_core_mask(
                image_id,
                medium_kernel_size=5,
                medium_iterations=1,
                eye_kernel_size=10,
                eye_iterations=1,
            )
            mask_v2 = erode_mask(mask_v2_core, iterations=1)

            mask_v3_core = build_semantic_core_mask(
                image_id,
                medium_kernel_size=5,
                medium_iterations=1,
                eye_kernel_size=25,
                eye_iterations=1,
            )
            mask_v3 = erode_mask(mask_v3_core, iterations=2)
            
            mask_v4_core = build_semantic_core_mask(
                image_id,
                medium_kernel_size=7,
                medium_iterations=1,
                eye_kernel_size=29,
                eye_iterations=1,
            )
            mask_v4 = erode_mask(mask_v4_core, iterations=3)

        except FileNotFoundError as e:
            print(f"Skipping {image_id}: {e}")
            continue

        cv2.imwrite(str(OUT_V1 / f"{image_id}.png"), mask_v1 * 255)
        cv2.imwrite(str(OUT_V2 / f"{image_id}.png"), mask_v2 * 255)
        cv2.imwrite(str(OUT_V3 / f"{image_id}.png"), mask_v3 * 255)
        cv2.imwrite(str(OUT_V4 / f"{image_id}.png"), mask_v4 * 255)


        cv2.imwrite(str(OVERLAY_V1 / f"{image_id}.png"), make_overlay(image, mask_v1))
        cv2.imwrite(str(OVERLAY_V2 / f"{image_id}.png"), make_overlay(image, mask_v2))
        cv2.imwrite(str(OVERLAY_V3 / f"{image_id}.png"), make_overlay(image, mask_v3))
        cv2.imwrite(str(OVERLAY_V4 / f"{image_id}.png"), make_overlay(image, mask_v4))

        print(f"Saved pilot outputs for {image_id}")


if __name__ == "__main__":
    main()