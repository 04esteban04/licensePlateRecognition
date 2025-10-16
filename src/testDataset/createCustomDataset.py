import os
import random
import shutil
import yaml
from pathlib import Path
import cv2
import numpy as np

# === CONFIGURACIÓN ===
SOURCE_DIR = "images_raw"   # Carpeta con tus imágenes base: char_0.jpg, char_A.jpg, etc.
OUTPUT_DIR = "../dataset/yoloDatasetRecognition"
VAL_SPLIT = 0.2
AUG_PER_IMAGE = 300         # Número de variaciones por imagen

# === CLASES ===
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

# === CREAR ESTRUCTURA DE CARPETAS ===
for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# === RECOLECTAR IMÁGENES ===
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(all_images)

split_idx = int(len(all_images) * (1 - VAL_SPLIT))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

# === FUNCIONES DE AUMENTO ===
def random_augment(img):
    """Aplica una combinación aleatoria de aumentos."""
    # Rotación
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brillo / contraste
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Desenfoque aleatorio
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Ruido
    if random.random() < 0.5:
        noise = np.random.normal(0, random.randint(5, 25), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Inversión de color aleatoria
    if random.random() < 0.2:
        img = cv2.bitwise_not(img)

    # Escalado / recorte leve
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        if scale > 1:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            img = img[start_y:start_y + h, start_x:start_x + w]
        else:
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)
            img = cv2.resize(img, (w, h))

    return img

# === PROCESAR Y AUMENTAR IMÁGENES ===
def process_images(image_list, split):
    for img_name in image_list:
        img_path = os.path.join(SOURCE_DIR, img_name)
        char = os.path.splitext(img_name)[0].upper().replace("CHAR_", "").replace("CHAR", "")

        if char not in CLASS_TO_ID:
            print(f"⚠️ Carácter desconocido: {char}")
            continue

        class_id = CLASS_TO_ID[char]

        # Leer imagen original
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ No se pudo leer {img_name}")
            continue

        for i in range(AUG_PER_IMAGE):
            aug_img = random_augment(img.copy())
            new_name = f"{os.path.splitext(img_name)[0]}_{i:03d}.jpg"
            dest_img = os.path.join(OUTPUT_DIR, f"{split}/images", new_name)
            cv2.imwrite(dest_img, aug_img)

            label_path = os.path.join(OUTPUT_DIR, f"{split}/labels", f"{os.path.splitext(new_name)[0]}.txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        print(f"✅ {img_name} → {AUG_PER_IMAGE} variaciones → clase {class_id}")

# === GENERAR DATASET ===
process_images(train_imgs, "train")
process_images(val_imgs, "val")

# === CREAR ARCHIVO data.yaml ===
data_yaml = {
    "train": f"./{OUTPUT_DIR}/train/images",
    "val": f"./{OUTPUT_DIR}/val/images",
    "nc": len(CLASSES),
    "names": CLASSES
}

with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("\n🎉 Dataset YOLO aumentado generado correctamente en:", OUTPUT_DIR)
