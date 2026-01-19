import os
import shutil
import random
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
source_dir = Path(r"C:\Users\user\fruit-vision\data\fruit_dataset")

output_base = Path(r"C:\Users\user\fruit-vision\data")
train_dir = output_base / "train"
val_dir   = output_base / "val"
test_dir  = output_base / "test"

# split ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)

# -----------------------------
# Clean old splits if exist
# -----------------------------
for folder in [train_dir, val_dir, test_dir]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

# -----------------------------
# Split logic
# -----------------------------
for class_folder in source_dir.iterdir():
    if not class_folder.is_dir():
        continue

    images = list(class_folder.glob("*"))
    random.shuffle(images)

    n = len(images)
    train_end = int(TRAIN_RATIO * n)
    val_end   = train_end + int(VAL_RATIO * n)

    splits = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:]
    }

    for split_name, files in splits.items():
        dest = output_base / split_name / class_folder.name
        dest.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, dest / f.name)

print("Dataset split complete!")
print(f"Train → {train_dir}")
print(f"Val   → {val_dir}")
print(f"Test  → {test_dir}")
