import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
import yaml
import matplotlib.pyplot as plt


# ---------------- CONFIG ---------------- #
OUTPUT_DIR = Path("../dataset/dataset_characters_yolo")
CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
NUM_IMAGES_PER_CLASS = 300

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    #cv2.FONT_HERSHEY_PLAIN,
]
FONT_SCALES = [1, 1.5, 2]
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

IMG_SIZE = 64  # width = height = 64px

random.seed(42)
np.random.seed(42)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
THICKNESS = 2


# ---------------- UTILITIES ---------------- #
def resetOutputDir():
    """Deletes existing dataset and recreates folder structure."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for subset in SPLIT_RATIOS.keys():
        (OUTPUT_DIR / subset / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / subset / "labels").mkdir(parents=True, exist_ok=True)
    print("📂 Dataset folder reset complete.")


def generateCharacterImage(char):
    """Generates a grayscale synthetic image of a single character."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)  # black background

    font = random.choice(FONTS)
    scale = random.choice(FONT_SCALES)
    thickness = random.randint(1, 3)

    #font = FONT
    #scale = FONT_SCALE
    #thickness = THICKNESS
    
    color = 255  # white text

    text_size = cv2.getTextSize(char, font, scale, thickness)[0]
    text_x = (IMG_SIZE - text_size[0]) // 2
    text_y = (IMG_SIZE + text_size[1]) // 2

    cv2.putText(img, char, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    # Threshold for binarization
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Random morphological operation
    # kernel = np.ones((2, 2), np.uint8)
    # if random.random() > 0.5:
    #     img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
    # else:
    #     img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    # # Add light noise
    # noise = np.random.randint(0, 20, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    # img_noisy = cv2.add(img_bin, noise)

    return cv2.resize(img_bin, (IMG_SIZE, IMG_SIZE))


def getBoundingBox(img):
    """Finds the bounding box (x_center, y_center, w, h) normalized for YOLO format."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.5, 0.5, 1.0, 1.0]  # default if none found

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Normalize for YOLO format
    x_center = (x + w / 2) / IMG_SIZE
    y_center = (y + h / 2) / IMG_SIZE
    w_norm = w / IMG_SIZE
    h_norm = h / IMG_SIZE
    return [x_center, y_center, w_norm, h_norm]


def saveImageAndLabel(img, subset, char, idx):
    """Saves image and corresponding YOLO annotation."""
    subset_dir = OUTPUT_DIR / subset
    image_dir = subset_dir / "images"
    label_dir = subset_dir / "labels"

    img_filename = f"{char}_{idx:04d}.png"
    label_filename = f"{char}_{idx:04d}.txt"

    img_path = image_dir / img_filename
    label_path = label_dir / label_filename

    # Save image
    cv2.imwrite(str(img_path), img)

    # Save YOLO label
    bbox = getBoundingBox(img)
    class_id = CHARACTERS.index(char)
    label_str = f"{class_id} " + " ".join(f"{x:.6f}" for x in bbox) + "\n"

    with open(label_path, "w") as f:
        f.write(label_str)


def generateDataset():
    """Generates the complete YOLO-ready synthetic dataset."""
    resetOutputDir()

    for char in CHARACTERS:
        print(f"Generating images for {char}...")
        total_images = [generateCharacterImage(char) for _ in range(NUM_IMAGES_PER_CLASS)]

        n_train = int(NUM_IMAGES_PER_CLASS * SPLIT_RATIOS["train"])
        n_val = int(NUM_IMAGES_PER_CLASS * SPLIT_RATIOS["val"])

        for i, img in enumerate(total_images):
            if i < n_train:
                subset = "train"
            elif i < n_train + n_val:
                subset = "val"
            else:
                subset = "test"
            saveImageAndLabel(img, subset, char, i)

    print("\n✅ YOLO dataset successfully generated at 'dataset_characters_yolo/'")


def createDataYaml():
    """Creates YOLO-compatible data.yaml file."""
    yaml_path = OUTPUT_DIR / "data.yaml"
    data = {
        "train": str(OUTPUT_DIR / "train" / "images"),
        "val": str(OUTPUT_DIR / "val" / "images"),
        "test": str(OUTPUT_DIR / "test" / "images"),
        "nc": len(CHARACTERS),
        "names": CHARACTERS,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"🧾 data.yaml file created at: {yaml_path}")


def showRandomSamples(num=10):
    """Displays random samples from the dataset with bounding boxes."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax in axes.flatten():
        subset = random.choice(list(SPLIT_RATIOS.keys()))
        images_path = OUTPUT_DIR / subset / "images"
        labels_path = OUTPUT_DIR / subset / "labels"

        img_file = random.choice(list(images_path.glob("*.png")))
        label_file = labels_path / (img_file.stem + ".txt")

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        if label_file.exists():
            with open(label_file) as f:
                data = f.readline().strip().split()
                if len(data) == 5:
                    _, x, y, bw, bh = map(float, data)
                    x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
                    x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (200,), 1)

        ax.imshow(img, cmap="gray")
        ax.set_title(f"{subset}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":
    generateDataset()
    createDataYaml()
    showRandomSamples()
