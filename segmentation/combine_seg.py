import cv2
import numpy as np
from pathlib import Path

INPUT_ROOT = Path("data")
MASK_ROOT = Path("data_masks")
OUTPUT_ROOT = Path("data_combined")

BREEDS = ["Bengal", "Calico", "Persian", "Siamese", "Sphynx"]
SPLITS = ["train", "validation", "test"]

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def combine_image_and_mask(img_path, mask_path, save_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"⚠️ Could not read {img_path} or its mask")
        return

    mask_norm = mask / 255.0
    mask_3c = np.repeat(mask_norm[:, :, None], 3, axis=2)

    fg = img[:, :, :3] * mask_3c

    bg = np.zeros_like(fg)

    final = fg + bg

    alpha = (mask_norm * 255).astype(np.uint8)
    final_rgba = np.dstack([final.astype(np.uint8), alpha])

    cv2.imwrite(str(save_path), final_rgba)
    print(f"✔ Saved combined: {save_path.name}")

def process_split(split):
    print(f"\n▶ Processing {split} split...")

    for breed in BREEDS:
        in_dir = INPUT_ROOT / split / breed
        mask_dir = MASK_ROOT / split / breed
        out_dir = OUTPUT_ROOT / split / breed

        ensure_dir(out_dir)

        images = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpeg"))

        for img_path in images:
            name = img_path.stem
            mask_path = mask_dir / f"{name}_mask.png"
            save_path = out_dir / f"{name}.png"

            if not mask_path.exists():
                print(f"⚠️ No mask for {img_path.name}")
                continue

            combine_image_and_mask(img_path, mask_path, save_path)

    print(f"✔ Completed {split}.")

if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)
    print("\n🎉 COMBINED DATASET CREATED SUCCESSFULLY!")





