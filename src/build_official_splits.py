from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "CelebA-HQ-to-CelebA-mapping.txt"
PARTITION_PATH = PROJECT_ROOT / "data" / "metadata" / "list_eval_partition.txt"
MASK_DIR = PROJECT_ROOT / "data" / "processed_masks" / "final_v4"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_OUT = SPLIT_DIR / "train_ids.txt"
VAL_OUT = SPLIT_DIR / "val_ids.txt"
TEST_OUT = SPLIT_DIR / "test_ids.txt"


def load_partition_map() -> dict[str, str]:
    partition_map = {}

    with open(PARTITION_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            celeba_filename, split_id = parts
            partition_map[celeba_filename] = split_id

    return partition_map


def load_hq_to_celeba_map() -> dict[str, str]:
    hq_to_celeba = {}

    with open(MAPPING_PATH, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split()

        if len(parts) < 3:
            continue

        hq_idx = parts[0].strip()
        celeba_file = parts[2].strip()

        hq_to_celeba[hq_idx] = celeba_file

    return hq_to_celeba


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    partition_map = load_partition_map()
    hq_to_celeba = load_hq_to_celeba_map()

    train_ids = []
    val_ids = []
    test_ids = []

    mask_paths = sorted(MASK_DIR.glob("*.png"))

    for mask_path in mask_paths:
        image_id = mask_path.stem
        celeba_file = hq_to_celeba.get(image_id)

        if celeba_file is None:
            print(f"Missing mapping for HQ id: {image_id}")
            continue

        split_id = partition_map.get(celeba_file)

        if split_id is None:
            print(f"Missing partition for CelebA file: {celeba_file}")
            continue

        if split_id == "0":
            train_ids.append(image_id)
        elif split_id == "1":
            val_ids.append(image_id)
        elif split_id == "2":
            test_ids.append(image_id)

    TRAIN_OUT.write_text("\n".join(train_ids) + "\n")
    VAL_OUT.write_text("\n".join(val_ids) + "\n")
    TEST_OUT.write_text("\n".join(test_ids) + "\n")

    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")


if __name__ == "__main__":
    main()