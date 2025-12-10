import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from preprocessing.config import SEGMENTED

if SEGMENTED:
    DATA_ROOT = Path("data_combined")  # segmented images
else:
    DATA_ROOT = Path("data")  


def create_loaders(batch_size=16, img_size=224):
    if SEGMENTED:     
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    train_dir = DATA_ROOT / "train"
    val_dir   = DATA_ROOT / "validation"
    test_dir  = DATA_ROOT / "test"

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds   = datasets.ImageFolder(val_dir,   transform=transform)
    test_ds  = datasets.ImageFolder(test_dir,  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_map = train_ds.class_to_idx
    return train_loader, val_loader, test_loader, class_map
