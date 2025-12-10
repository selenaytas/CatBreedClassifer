import os
import cv2
import numpy as np
from pathlib import Path
from segmentation.u2net_seg import u2net_segment

INPUT_ROOT = Path("data")             # Orijinal dataset
OUTPUT_ROOT = Path("data_masks")      # Maskelerin kaydedileceği yer

BREEDS = ["Bengal", "Calico", "Persian", "Siamese", "Sphynx"]
SPLITS = ["train", "validation", "test"]   # ⬅ val yerine validation

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    print(f"\n🔹 Processing {split} split...")

    for breed in BREEDS:
        in_dir = INPUT_ROOT / split / breed
        out_dir = OUTPUT_ROOT / split / breed
        ensure_dir(out_dir)

        images = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.jpeg")) + list(in_dir.glob("*.png"))

        for img_path in images:
            mask = u2net_segment(str(img_path))  # 0–255 maske

            # Tam siyah-beyaz maske
            mask_bin = (mask > 128).astype(np.uint8) * 255

            save_path = out_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(save_path), mask_bin)

            print("✔ Mask saved:", save_path.name)

print("\n🎉 ALL MASKS GENERATED → data_masks klasörüne bak")

