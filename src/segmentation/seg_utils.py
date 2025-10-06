""" import cv2
import numpy as np
from pathlib import Path

def segmentCharacters(platePath, outputDir="outputs/characters"):
    
    #Segment characters from a cropped license plate image.
    

    plateImg = cv2.imread(str(platePath))
    if plateImg is None:
        raise FileNotFoundError(f"Image not found: {platePath}")

    gray = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Binarize (convert to black & white)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (potential character regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Output directory
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    # Filter & crop characters
    charIndex = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out noise / non-character regions
        if h > 0.5 * plateImg.shape[0] and 0.2 * plateImg.shape[1] > w > 10:
            charCrop = plateImg[y:y+h, x:x+w]
            savePath = Path(outputDir) / f"{Path(platePath).stem}_char{charIndex+1}.jpg"
            cv2.imwrite(str(savePath), charCrop)
            charIndex += 1

    print(f"✅ {charIndex} characters saved to {outputDir}")
 """


import cv2
from pathlib import Path
import numpy as np

def convertToGrayscale(inputDir="outputs/crops", outputDir="outputs/characters_gray"):
    """
    Lee todas las imágenes de inputDir y las guarda en escala de grises en outputDir.
    """

    inputDir = Path(inputDir)
    outputDir = Path(outputDir)

    image_paths = sorted(inputDir.glob("*.*"))  # lee todas las imágenes
    if not image_paths:
        print(f"No se encontraron imágenes en {inputDir}")
        return

    for i, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  No se pudo leer la imagen: {img_path}")
            continue

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste
        gray = cv2.equalizeHist(gray)

        # Binarización usando Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        savePath = outputDir / f"{img_path.stem}_bin.jpg"
        cv2.imwrite(str(savePath), binary)

        print(f"✅ Imagen {i}: {img_path.name} convertida y binarizada -> {savePath}")




def segmentByColumns(inputDir="outputs/characters", outputDir="outputs/characters_segmented"):
    """
    Segmenta caracteres de cada imagen binarizada usando proyección vertical.
    Guarda cada carácter como una imagen individual en outputDir.
    """

    inputDir = Path(inputDir)
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(inputDir.glob("*.*"))
    if not image_paths:
        print(f"No se encontraron imágenes en {inputDir}")
        return

    for i, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ No se pudo leer la imagen: {img_path}")
            continue

        # --- Proyección vertical ---
        projection = np.sum(img == 255, axis=0)  # contar píxeles blancos por columna
        norm_proj = projection / np.max(projection)

        # Umbral para detectar columnas con caracteres
        threshold = 0.1  # ajustar si es necesario
        binary_proj = (norm_proj > threshold).astype(np.uint8)

        # Detectar segmentos continuos
        segments = []
        start = None
        for j, val in enumerate(binary_proj):
            if val and start is None:
                start = j
            elif not val and start is not None:
                end = j
                if end - start > 3:  # descarta segmentos muy pequeños
                    segments.append((start, end))
                start = None
        if start is not None:
            segments.append((start, len(binary_proj)))

        # --- Guardar caracteres segmentados ---
        plate_name = img_path.stem
        plate_output_dir = outputDir / plate_name
        plate_output_dir.mkdir(parents=True, exist_ok=True)

        for k, (x_start, x_end) in enumerate(segments):
            char_crop = img[:, x_start:x_end]
            save_path = plate_output_dir / f"{plate_name}_char{k+1}.jpg"
            cv2.imwrite(str(save_path), char_crop)

        print(f"✅ {len(segments)} caracteres segmentados de {img_path.name} en {plate_output_dir}")
