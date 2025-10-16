import shutil
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from PIL import Image


# ---------------- CONFIG ---------------- #
MODEL_NAME = "yolo11n.pt"  # lightweight YOLOv11 model
MODEL_DIR = Path("models/yoloRecognition")
DATASET_DIR = Path("../dataset/dataset_characters_yolo")
OUTPUT_DIRS = ["models/yoloRecognition"]
EPOCHS = 100
IMG_SIZE = 64


# ---------------- UTILITIES ---------------- #
def cleanDirectories(dirs):
    """Remove specified directories if they exist."""
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"🧹 Removing directory: {path}")
            shutil.rmtree(path)


def loadModel(modelName=MODEL_NAME, saveDir=MODEL_DIR):
    """Load YOLO model from local storage or download if not found."""

    Path(saveDir).mkdir(parents=True, exist_ok=True)
    modelPath = Path(saveDir) / modelName

    if modelPath.exists():
        print(f"📦 Loading local model: {modelPath}")
        return YOLO(str(modelPath))

    else:
        print(f"⬇️ Downloading model: {modelName}")
        model = YOLO(modelName)

        if Path(modelName).exists():
            shutil.move(modelName, modelPath)
            print(f"💾 Model saved to: {modelPath}")

    return model


def prepareDataset(datasetDir=DATASET_DIR):
    """Prepare YOLO dataset and fix data.yaml paths."""
    datasetPath = datasetDir.expanduser().resolve()

    yamlFiles = list(datasetPath.rglob("data.yaml"))
    if not yamlFiles:
        raise FileNotFoundError(f"❌ data.yaml not found in {datasetPath}")

    dataYaml = yamlFiles[0]
    print(f"📁 Using dataset: {dataYaml}")

    with open(dataYaml, "r") as f:
        data = yaml.safe_load(f)

    # Ensure absolute paths in YAML
    data["train"] = str((datasetPath / "train" / "images").resolve())
    data["val"] = str((datasetPath / "val" / "images").resolve())
    data["test"] = str((datasetPath / "test" / "images").resolve())

    with open(dataYaml, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print("✅ Dataset paths fixed.")
    return str(dataYaml), data


def trainModel(model, data, epochs=EPOCHS, imgsz=IMG_SIZE, project=MODEL_DIR, name="train", workers=2):
    """Train YOLO model."""
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Training on: {'GPU' if device != 'cpu' else 'CPU'}")

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        workers=2,
        exist_ok=True
    )
    print("✅ Training complete.")
    return results


def evaluateModel(model, data, project=MODEL_DIR, name="validation"):
    """Evaluate YOLO model."""
    print("📊 Evaluating model...")
    results = model.val(data=data, project=str(project), name=name)
    print("✅ Validation complete.")
    return results


def exportModel(model, exportFormat="onnx", project=MODEL_DIR, name="export"):
    """Export YOLO model to given format (default ONNX)."""
    print(f"📦 Exporting model to {exportFormat.upper()} format...")
    exportPath = model.export(format=exportFormat, project=str(project), name=name)
    print(f"✅ Model exported to: {exportPath}")
    return Path(exportPath)


def predictImage(model, imagePath, project="outputs", name="predictChar", show=True, crop=True):
    """Run inference on a single image, save results, and print predicted labels."""
    print(f"🔍 Predicting: {imagePath}")
    
    # Run inference
    results = model.predict(source=imagePath, project=project, name=name, save=True, verbose=False)
    result = results[0]  # take first result (since source is one image)
    
    # Show prediction
    if show:
        result.show()

    print(f"✅ Predictions saved to: {Path(project) / name}")

    # ---- PRINT LABELS ---- #
    if result.boxes is not None and len(result.boxes) > 0:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        print("\n🔠 Predicted Characters:")
        for cls_id, conf in zip(classes, confidences):
            label = model.names[cls_id]
            print(f"  - {label} (confidence: {conf:.2f})")
    else:
        print("⚠️ No characters detected.")

    # ---- OPTIONAL: Save cropped detections ---- #
    if crop:
        cropDetections(results, saveDir=Path(project) / "crops")

    return results


def cropDetections(results, saveDir="outputs/crops"):
    """Crop detected characters from results and save."""
    savePath = Path(saveDir)
    savePath.mkdir(parents=True, exist_ok=True)

    for r in results:
        imgPath = Path(r.path)
        img = Image.open(imgPath).convert("RGB")

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cropped = img.crop((x1, y1, x2, y2))
            cropFileName = savePath / f"{imgPath.stem}_{j}.png"
            cropped.save(cropFileName)
            print(f"🖼️ Saved cropped image: {cropFileName}")


# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":
    
    cleanDirectories(OUTPUT_DIRS)

    # Load YOLO model
    model = loadModel()

    # Load dataset YAML
    dataYamlPath, data = prepareDataset()

    # Train model
    results = trainModel(model, dataYamlPath)

    # Validate model
    evaluateModel(model, dataYamlPath)

    # (Optional) Export model
    exportModel(model)
    
    print("\n🚀 All done! Your YOLO model is trained and ready to use.")
   
    # Test prediction on a sample image    
    model = loadModel("../models/yoloRecognition/train/weights/best.pt")
    
    prediction = predictImage(model, imagePath="../dataset/dataset_characters_yolo/test/images/A_0280.png")
    print(prediction)
