from enum import Enum

SEGMENTED = True  

class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class DataPreprocessingConfig:
    RAW_DATA_DIR = "/Users/selenaytas/Desktop/keditür/images"
    MODEL_DATA_DIR = "/Users/selenaytas/Desktop/CatBreedsDatasetProject/data"

    SPLIT_RATIOS = {
        DatasetSplit.TRAIN : 0.8,
        DatasetSplit.VALIDATION : 0.1,
        DatasetSplit.TEST : 0.1
    }

    BREED_ALIAS_MAP = {
        "Bengal": ["Bengal"],
        "Calico": ["Calico"],
        "Persian": ["Persian"],
        "Siamese": ["Siamese"],
        "Sphynx": ["Sphynx", "Sphynx - Hairless Cat"]
    }
