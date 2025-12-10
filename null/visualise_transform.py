from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch


img_size = 224

transform = transforms.Compose([
    transforms.Resize((img_size, img_size), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_path = "/Users/selenaytas/Desktop/CatBreedClassification/data_combined/train/Bengal/31826146_5173.png"   
img = Image.open(img_path).convert("RGB")

img_transformed = transform(img)

plt.imshow(img_transformed.permute(1, 2, 0))   # convert CHW → HWC
plt.title("Transformed (Resized) Image")
plt.axis("off")
plt.show()
