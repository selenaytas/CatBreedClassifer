import torch
import torch.nn as nn
from model.efficientnet import get_efficientnet_model
from model.dataset import create_loaders
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def evaluate_experiment(experiment_name):
    print(f"\n📌 Evaluating: {experiment_name}\n")

    model_path = f"experiments/{experiment_name}/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model bulunamadı: {model_path}")

    model = get_efficientnet_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    _, _, test_loader, class_map = create_loaders()
    inv_map = {v: k for k, v in class_map.items()}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))

    # === GRAFİK ===
    plt.figure(figsize=(14, 5))

    # Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=inv_map.values(),
                yticklabels=inv_map.values())
    plt.title("Confusion Matrix")

    # Precision / Recall / F1 / Accuracy
    plt.subplot(1, 2, 2)
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision (macro): {precision:.4f}\n"
        f"Recall (macro): {recall:.4f}\n"
        f"F1-score (macro): {f1:.4f}\n\n"
        f"Experiment: {experiment_name}"
    )
    plt.text(0.1, 0.5, metrics_text, fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"✔ Tamamlandı → {experiment_name}")

if __name__ == "__main__":
    #evaluate_experiment("exp8_adamw_lr5e-4_bs16_ep100")  # Burada değerlendirmek istediğiniz deneyin adını girin
    evaluate_experiment("exp_with_seg")