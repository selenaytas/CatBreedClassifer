import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.dataset import create_loaders
from model.efficientnet import get_efficientnet_model
from preprocessing.config import SEGMENTED

if SEGMENTED:
    EXP_NAME   = "exp_with_seg"  # her deneyde bunu değiştir
else:
    EXP_NAME   = "exp_without_seg"  # her deneyde bunu değiştir
EPOCHS     = 22
BATCH_SIZE = 16
LR         = 5e-4
# =======================================

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

# Experiments klasörü
EXP_DIR = os.path.join("experiments", EXP_NAME)
os.makedirs(EXP_DIR, exist_ok=True)

# Dataset
train_loader, val_loader, test_loader, class_map = create_loaders(batch_size=BATCH_SIZE)
print("Classes:", class_map)

# Model
model = get_efficientnet_model(num_classes=len(class_map))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

best_val_acc = 0.0

# Log dosyası
log_path = os.path.join(EXP_DIR, "training_log.csv")
with open(log_path, "w") as f:
    f.write("epoch,train_loss,train_acc,val_acc\n")

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = correct / total
    print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")
    return total_loss, train_acc


def validate(epoch):
    global best_val_acc
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = correct / total if total > 0 else 0.0
    print(f"[Epoch {epoch}] Validation Acc: {val_acc:.4f}")

    # En iyi modeli kaydet
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = os.path.join(EXP_DIR, "best_model.pth")
        torch.save(model.state_dict(), model_path)
        print("🔥 New best model saved:", model_path)

    return val_acc


# ========= TRAIN LOOP =========
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    val_acc = validate(epoch)

    # Log'a yaz
    with open(log_path, "a") as f:
        f.write(f"{epoch},{train_loss},{train_acc},{val_acc}\n")

print("🎉 Training complete!")
print("Best val acc:", best_val_acc)
print("Logs saved to:", log_path)

