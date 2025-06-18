import os
import random
from pathlib import Path

def create_split_files(processed_dir: str, splits_dir: str,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)  # Setting the seed

    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)

    splits_path.mkdir(parents=True, exist_ok=True)

    patient_dirs = [d.name for d in processed_path.iterdir() if d.is_dir()]
    patient_dirs.sort()
    random.shuffle(patient_dirs)

    total = len(patient_dirs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train = patient_dirs[:n_train]
    val = patient_dirs[n_train:n_train + n_val]
    test = patient_dirs[n_train + n_val:]

    with open(splits_path / "train.txt", "w") as f:
        f.writelines(f"{x}\n" for x in train)

    with open(splits_path / "val.txt", "w") as f:
        f.writelines(f"{x}\n" for x in val)

    with open(splits_path / "test.txt", "w") as f:
        f.writelines(f"{x}\n" for x in test)

    print(f"Split complete: {total} total")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"
    splits_dir = base_dir / "data" / "splits"

    create_split_files(processed_dir, splits_dir)
