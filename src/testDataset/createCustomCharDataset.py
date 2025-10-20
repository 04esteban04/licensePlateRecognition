import os
import random
import yaml
from pathlib import Path
import cv2
import numpy as np


DEFAULT_SOURCE_DIR = "../assets/charImages"   # Folder with base character images (e.g., char_A.jpg)
DEFAULT_OUTPUT_DIR = "../dataset/yoloCharDataset"
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SAMPLES_PER_IMAGE = 300     # Number of samples generated per base image
DEFAULT_SEED = None                 # Optional seed for reproducibility

# Define character classes: 0–9 and A–Z
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}


def setupOutputDirs(outputDir):
    """
    Creates the necessary directory structure for YOLO training.
    """
    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        Path(outputDir, sub).mkdir(parents=True, exist_ok=True)
    
    print(f"\nDirectory structure created under: {outputDir}\n")


def createSample(img):
    """
    Applies a random combination of transformations to create a unique sample.
    This simulates variability across character appearances.
    """
    h, w = img.shape[:2]

    # Random rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness/contrast adjustment
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random Gaussian blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Add random Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, random.randint(5, 25), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random color inversion
    if random.random() < 0.2:
        img = cv2.bitwise_not(img)

    # Random scaling and cropping/padding
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        newW, newH = int(w * scale), int(h * scale)
        img = cv2.resize(img, (newW, newH))

        if scale > 1:
            startX = (newW - w) // 2
            startY = (newH - h) // 2
            img = img[startY:startY + h, startX:startX + w]
        else:
            padX = (w - newW) // 2
            padY = (h - newH) // 2
            img = cv2.copyMakeBorder(img, padY, padY, padX, padX,
                                     cv2.BORDER_CONSTANT, value=255)
            img = cv2.resize(img, (w, h))

    return img


def processImages(sourceDir, imageList, outputDir, split, samplesPerImage):
    """
    Processes base images to generate a specified number of samples for each,
    and saves both images and YOLO labels.
    """
    for imgName in imageList:
        imgPath = Path(sourceDir) / imgName
        char = Path(imgName).stem.upper().replace("CHAR_", "").replace("CHAR", "")

        if char not in CLASS_TO_ID:
            print(f"Unknown character: {char}")
            continue

        classId = CLASS_TO_ID[char]

        # Read original image
        img = cv2.imread(str(imgPath))
        if img is None:
            print(f"Could not read {imgName}")
            continue

        for i in range(samplesPerImage):
            sampleImg = createSample(img.copy())

            newName = f"{Path(imgName).stem}_{i:03d}.jpg"
            destImg = Path(outputDir) / f"{split}/images" / newName
            labelPath = Path(outputDir) / f"{split}/labels" / f"{Path(newName).stem}.txt"

            # Save sample image
            cv2.imwrite(str(destImg), sampleImg)

            # Save YOLO label (centered bounding box)
            with open(labelPath, "w") as f:
                f.write(f"{classId} 0.5 0.5 1.0 1.0\n")

        print(f"    {imgName} \t → {samplesPerImage} samples \t → class {classId}")


def createDataYaml(outputDir):
    """
    Creates YOLO-compatible data.yaml file.
    """
    yamlPath = Path(outputDir) / "data.yaml"
    data = {
        "train": f"./{outputDir}/train/images",
        "val": f"./{outputDir}/val/images",
        "nc": len(CLASSES),
        "names": CLASSES
    }

    with open(yamlPath, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\nFile data.yaml successfully created at: {yamlPath}\n")


def generateYoloCharDataset(sourceDir=DEFAULT_SOURCE_DIR,
                        outputDir=DEFAULT_OUTPUT_DIR,
                        valSplit=DEFAULT_VAL_SPLIT,
                        samplesPerImage=DEFAULT_SAMPLES_PER_IMAGE,
                        seed=DEFAULT_SEED):
    """
    Main function to create a YOLO-formatted dataset from a base character set.

    Args:
        sourceDir (str): Directory containing the base character images.
        outputDir (str): Directory to save the generated dataset.
        valSplit (float): Validation split ratio.
        samplesPerImage (int): Number of samples generated per image.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n Using fixed random seed: {seed}\n")

    setupOutputDirs(outputDir)

    allImages = [f for f in os.listdir(sourceDir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    random.shuffle(allImages)

    # splitIdx = int(len(allImages) * (1 - valSplit))
    # trainImgs = allImages[:splitIdx]
    # valImgs = allImages[splitIdx:]

    # Balanced per-class split (ensures each class appears in train and val)
    trainImgs, valImgs = [], []
    
    for cls in CLASSES:
        clsImgs = [f for f in allImages if f.upper().startswith(f"CHAR_{cls}")]
    
        if len(clsImgs) == 0:
            continue
    
        random.shuffle(clsImgs)
        splitIdx = max(1, int(len(clsImgs) * (1 - valSplit)))
        trainImgs += clsImgs[:splitIdx]
        valImgs += clsImgs[splitIdx:]


    print(f"\nTotal base images: {len(allImages)}")
    print(f"   Train: {len(trainImgs)} | Validation: {len(valImgs)}\n")

    processImages(sourceDir, trainImgs, outputDir, "train", samplesPerImage)
    processImages(sourceDir, valImgs, outputDir, "val", samplesPerImage)

    # Create YOLO data.yaml
    createDataYaml(outputDir)


if __name__ == "__main__":

    generateYoloCharDataset(seed=42)

    print("\nYOLO character dataset generated successfully!\n")
