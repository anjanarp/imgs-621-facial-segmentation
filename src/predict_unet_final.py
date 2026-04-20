from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images" / "celeba"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_PATH = PROJECT_ROOT / "outputs" / "unet_final" / "models" / "best_unet_final.pt"
PRED_DIR = PROJECT_ROOT / "outputs" / "unet_final" / "predictions"

VAL_IDS_PATH = SPLIT_DIR / "val_ids.txt"
TEST_IDS_PATH = SPLIT_DIR / "test_ids.txt"

IMG_SIZE = 256
BATCH_SIZE = 8


def load_ids(txt_path: Path) -> list[str]:
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class InferenceDataset(Dataset):
    def __init__(self, image_dir: Path, image_ids: list[str], img_size: int = 256):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        original_h, original_w = image.shape[:2]

        image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_chw = np.transpose(image_rgb, (2, 0, 1))

        image_tensor = torch.tensor(image_chw, dtype=torch.float32)

        return image_tensor, image_id, original_h, original_w


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(device: torch.device) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model.to(device)


def save_predictions_for_split(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
) -> None:
    split_dir = PRED_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, image_ids, original_hs, original_ws) in enumerate(loader, start=1):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(image_ids)):
                pred_small = probs[i, 0]
                pred_bin = (pred_small > 0.5).astype(np.uint8) * 255

                original_h = int(original_hs[i])
                original_w = int(original_ws[i])

                pred_full = cv2.resize(
                    pred_bin,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                save_path = split_dir / f"{image_ids[i]}.png"
                cv2.imwrite(str(save_path), pred_full)

            if batch_idx % 100 == 0:
                print(f"  {split_name}: processed batch {batch_idx}/{len(loader)}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {MODEL_PATH}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")

    model = build_model(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    val_ids = load_ids(VAL_IDS_PATH)
    test_ids = load_ids(TEST_IDS_PATH)

    val_dataset = InferenceDataset(IMAGE_DIR, val_ids, img_size=IMG_SIZE)
    test_dataset = InferenceDataset(IMAGE_DIR, test_ids, img_size=IMG_SIZE)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Val images: {len(val_ids)}")
    save_predictions_for_split(model, val_loader, "val", device)

    print(f"Test images: {len(test_ids)}")
    save_predictions_for_split(model, test_loader, "test", device)

    print("Done saving U-Net predictions.")


if __name__ == "__main__":
    main()