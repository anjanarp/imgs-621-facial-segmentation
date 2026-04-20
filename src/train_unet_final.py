from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images" / "celeba"
MASK_DIR = PROJECT_ROOT / "data" / "processed_masks" / "final_v4"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_IDS_PATH = SPLIT_DIR / "train_ids.txt"
VAL_IDS_PATH = SPLIT_DIR / "val_ids.txt"
TEST_IDS_PATH = SPLIT_DIR / "test_ids.txt"

OUT_DIR = PROJECT_ROOT / "outputs" / "unet_final"
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"

IMG_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3


def load_ids(txt_path: Path) -> list[str]:
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


class SkinDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_ids: list[str], img_size: int = 256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        image_path = self.image_dir / f"{image_id}.jpg"
        mask_path = self.mask_dir / f"{image_id}.png"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image_tensor, mask_tensor, image_id
    
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


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def compute_batch_iou_and_dice(preds: torch.Tensor, masks: torch.Tensor) -> tuple[float, float]:
    preds_bin = (preds > 0.5).float()

    intersection = (preds_bin * masks).sum(dim=(1, 2, 3))
    pred_area = preds_bin.sum(dim=(1, 2, 3))
    mask_area = masks.sum(dim=(1, 2, 3))
    union = pred_area + mask_area - intersection

    iou = (intersection / (union + 1e-8)).mean().item()
    dice = ((2 * intersection) / (pred_area + mask_area + 1e-8)).mean().item()

    return iou, dice

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dice_loss_fn,
    bce_loss_fn,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        loss = dice_loss_fn(probs, masks) + bce_loss_fn(probs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    dice_loss_fn,
    bce_loss_fn,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()

    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            loss = dice_loss_fn(probs, masks) + bce_loss_fn(probs, masks)
            iou, dice = compute_batch_iou_and_dice(probs, masks)

            running_loss += loss.item()
            running_iou += iou
            running_dice += dice

    mean_loss = running_loss / len(loader)
    mean_iou = running_iou / len(loader)
    mean_dice = running_dice / len(loader)

    return mean_loss, mean_iou, mean_dice

def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = load_ids(TRAIN_IDS_PATH)
    val_ids = load_ids(VAL_IDS_PATH)

    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")

    train_dataset = SkinDataset(IMAGE_DIR, MASK_DIR, train_ids, img_size=IMG_SIZE)
    val_dataset = SkinDataset(IMAGE_DIR, MASK_DIR, val_ids, img_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(device)

    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_iou = -1.0
    best_model_path = MODEL_DIR / "best_unet_final.pt"

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            dice_loss_fn=dice_loss_fn,
            bce_loss_fn=bce_loss_fn,
            device=device,
        )

        val_loss, val_iou, val_dice = validate_one_epoch(
            model=model,
            loader=val_loader,
            dice_loss_fn=dice_loss_fn,
            bce_loss_fn=bce_loss_fn,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_dice={val_dice:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved new best model to: {best_model_path}")


if __name__ == "__main__":
    main()