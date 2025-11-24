import os, random, shutil
from pathlib import Path
from config import DataPreprocessingConfig, DatasetSplit

def split_data():
    for split in DatasetSplit:
        for breed in DataPreprocessingConfig.BREED_ALIAS_MAP.keys():
            (Path(DataPreprocessingConfig.MODEL_DATA_DIR) / split.value / breed).mkdir(parents=True, exist_ok=True)

    for breed, aliases in DataPreprocessingConfig.BREED_ALIAS_MAP.items():
        imgs = []
        for alias in aliases:
            path = Path(DataPreprocessingConfig.RAW_DATA_DIR) / alias
            if path.exists():
                imgs.extend(
                    list(path.glob("*.jpg")) +
                    list(path.glob("*.jpeg")) +
                    list(path.glob("*.png"))
                )
        if not imgs:
            print(f"⚠️ No images found for {breed}")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        train_r = DataPreprocessingConfig.SPLIT_RATIOS[DatasetSplit.TRAIN]
        val_r = DataPreprocessingConfig.SPLIT_RATIOS[DatasetSplit.VALIDATION]
        train_end = int(train_r * n)
        val_end = int((train_r + val_r) * n)

        for i, img in enumerate(imgs):
            if i < train_end:
                split = DatasetSplit.TRAIN.value
            elif i < val_end:
                split = DatasetSplit.VALIDATION.value
            else:
                split = DatasetSplit.TEST.value
            shutil.copy(img, Path(DataPreprocessingConfig.MODEL_DATA_DIR) / split / breed / img.name)

        print(f"{breed} done → total {n} images")
    print("Split complete! Check:", DataPreprocessingConfig.MODEL_DATA_DIR)


if __name__ == "__main__":
    split_data()
