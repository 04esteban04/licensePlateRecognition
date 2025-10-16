import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path
import yaml
import matplotlib.pyplot as plt


# ---------------- CONFIG ---------------- #
OUTPUT_DIR = Path("../dataset/dataset_characters")
CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
NUM_IMAGES_PER_CLASS = 300

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]
FONT_SCALES = [1, 1.5, 2]

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


# ---------------- UTILITIES ---------------- #
def resetOutputDir():
    """Deletes existing dataset and recreates directory structure."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for subset in SPLIT_RATIOS.keys():
        (OUTPUT_DIR / subset).mkdir(parents=True, exist_ok=True)
    print("📂 Dataset folder reset complete.")


def generateCharacterImage(char):
    """Generates a single synthetic grayscale image for a given character."""
    img = np.zeros((64, 64), dtype=np.uint8)  # black background

    font = random.choice(FONTS)
    scale = random.choice(FONT_SCALES)
    thickness = random.randint(1, 3)
    color = 255  # white text

    text_size = cv2.getTextSize(char, font, scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2

    cv2.putText(img, char, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    # Threshold for binarization
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Random morphological operation
    kernel = np.ones((2, 2), np.uint8)
    if random.random() > 0.5:
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
    else:
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    # Add light noise
    noise = np.random.randint(0, 20, (64, 64), dtype=np.uint8)
    img_noisy = cv2.add(img_bin, noise)

    return cv2.resize(img_noisy, (64, 64))


def saveImage(img, subset, char, idx):
    """Saves an image into the appropriate subset/class folder."""
    folder = OUTPUT_DIR / subset / char
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"{char}_{idx:04d}.png"
    cv2.imwrite(str(filename), img)


def generateDataset():
    """Generates synthetic dataset split into train, val, and test folders."""
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
            saveImage(img, subset, char, i)

    print("\n✅ Dataset successfully generated at 'dataset_characters/'")


def createDataYaml():
    """Creates YOLO-compatible data.yaml file."""
    yaml_path = OUTPUT_DIR / "data.yaml"
    data = {
        "train": str(OUTPUT_DIR / "train"),
        "val": str(OUTPUT_DIR / "val"),
        "test": str(OUTPUT_DIR / "test"),
        "nc": len(CHARACTERS),
        "names": CHARACTERS,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"🧾 data.yaml file created at: {yaml_path}")


def countImagesPerClass():
    """Prints number of images per class in each subset."""
    print("\n📊 Dataset Summary:")
    for subset in SPLIT_RATIOS.keys():
        print(f"\nSubset: {subset.upper()}")
        total = 0
        for c in CHARACTERS:
            class_path = OUTPUT_DIR / subset / c
            count = len(list(class_path.glob("*.png")))
            print(f"  {c}: {count} images")
            total += count
        print(f"  ➤ Total images in {subset}: {total}")


def showRandomSamples(num=10):
    """Displays random images from the dataset."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax in axes.flatten():
        subset = random.choice(list(SPLIT_RATIOS.keys()))
        c = random.choice(CHARACTERS)
        class_path = OUTPUT_DIR / subset / c
        if not class_path.exists():
            continue
        img_path = random.choice(list(class_path.glob("*.png")))
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{subset}/{c}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":
    generateDataset()
    createDataYaml()
    countImagesPerClass()
    showRandomSamples()
