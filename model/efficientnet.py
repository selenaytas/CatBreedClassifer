import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_efficientnet_model(num_classes=5):
    """
    EfficientNet-B0 model
    """
    # Load ImageNet pretrained
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze base layers (istenirse açılabilir)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


