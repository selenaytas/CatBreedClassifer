import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim

from model.dataset import create_loaders
from model.efficientnet import get_efficientnet_model

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

# Hyperparameters
LR = 5e-4
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_THRESHOLD = 0.01       # Train ve Val accuracy birbirine çok yaklaşırsa durdur
PATIENCE = 5                          # 5 epoch boyunca gelişme olmazsa durdur

# Load loaders
train_loader, val_loader, test_loader, class_map = create_loaders()
print("Classes:", class_map)

# Model
model = get_efficientnet_model(num_classes=len(class_map)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0
patience_counter = 0

def train_one_epoch():
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = correct / total
    return total_loss, train_acc

def validate():
    global best_val_acc
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = correct / total
    return val_acc


# -------------------------
# 🚀 TRAINING LOOP
# -------------------------
for epoch in range(1, MAX_EPOCHS+1):
    print(f"\n===== Epoch {epoch}/{MAX_EPOCHS} =====")

    train_loss, train_acc = train_one_epoch()
    val_acc = validate()

    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {train_loss:.4f}")

    # 🔥 Best model kaydet
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_final_model.pth")
        print("💾 Best model saved!")
        patience_counter = 0
    else:
        patience_counter += 1

    # 🔥 Early Stopping – Train ve Val accuracy çok benzerse durdur
    acc_diff = abs(train_acc - val_acc)
    if acc_diff < EARLY_STOPPING_THRESHOLD:
        print(f"🛑 Early stopping: Train-Val accuracy difference {acc_diff:.4f} < {EARLY_STOPPING_THRESHOLD}")
        break

    # 🔥 Patience – 5 epoch boyunca gelişim yoksa durdur
    if patience_counter >= PATIENCE:
        print("🛑 Early stopping: No improvement for 5 epochs.")
        break

print("\n🎉 Training finished! Best model = best_final_model.pth")
