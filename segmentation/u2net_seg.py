import cv2
import numpy as np
from rembg import remove

def u2net_segment(image_path):
    # Orijinal resmi oku
    with open(image_path, "rb") as f:
        data = f.read()

    # rembg çıktısı RGBA png'dir
    result = remove(data)

    # Byte'tan array'e çevir
    img = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)

    # Alpha kanalı al (arkaplan 0, nesne 255)
    alpha = img[:, :, 3]

    # 1️⃣ Normalize et
    alpha_norm = alpha / 255.0

    # 2️⃣ Threshold (tam siyah–beyaz)
    mask = (alpha_norm > 0.2).astype(np.uint8) * 255

    return mask


