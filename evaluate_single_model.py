import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from model.dataset import create_loaders
from model.efficientnet import get_efficientnet_model

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "best_final_model.pth"

def evaluate(model_path):
    print(f"📌 Model yükleniyor: {model_path}")

    # dataset yükle
    train_loader, val_loader, test_loader, class_map = create_loaders()
    idx_to_class = {v: k for k, v in class_map.items()}

    # model oluştur
    model = get_efficientnet_model(num_classes=len(class_map))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # METRİKLER
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\n📊 TEST METRİKLERİ")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(class_map.keys()),
                yticklabels=list(class_map.keys()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True, help="best_model.pth dosyasının yolu")
    # args = parser.parse_args()

    evaluate(MODEL_PATH)
