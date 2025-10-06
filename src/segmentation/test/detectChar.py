from pathlib import Path
from segmentation.seg_utils import convertToGrayscale, segmentByColumns
import cv2
import numpy as np

if __name__ == "__main__":

    
    """ 
    # Leer la imagen
    img = cv2.imread(str(Path("outputs/crops/plate_plate1.jpg").resolve()))
    #img = cv2.imread(str(Path("outputs/crops/test_plate1.jpg").resolve()))

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarizar (umbral adaptativo para resaltar caracteres)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 10)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copia de la imagen para dibujar
    output = img.copy()

    # Filtrar y dibujar rectángulos
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtrar por tamaño (ajusta estos valores)
        if 10 < w < 100 and 20 < h < 200:
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 1.0:  # caracteres suelen ser más altos que anchos
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar resultado
    cv2.imshow("Caracteres detectados", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    # Leer imagen
    #img = cv2.imread(str(Path("outputs/crops/plate_plate1.jpg").resolve()))
    img = cv2.imread(str(Path("outputs/crops/test_plate1.jpg").resolve()))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reducir ruido
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Binarizar (umbral adaptativo)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 10)

    # Morfología: cerrar huecos y unir fragmentos finos
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # Duplicar caracteres finos para que no se rompan
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    filtered_contours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Tamaño permisivo para caracteres pequeños y guiones
        if 6 < w < 150 and 18 < h < 200:
            aspect_ratio = w / float(h)
            if 0.08 < aspect_ratio < 1.0:  # más delgado para guiones y 'I'
                area = cv2.contourArea(cnt)
                if area > 20:  # eliminar ruido extremadamente pequeño
                    filtered_contours.append((x, y, w, h))

    # Ordenar contornos de izquierda a derecha
    filtered_contours = sorted(filtered_contours, key=lambda b: b[0])

    # Dibujar rectángulos sobre los caracteres detectados
    for x, y, w, h in filtered_contours:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    imgResize = cv2.resize(output, (800, 200), interpolation=cv2.INTER_AREA)


    cv2.imshow("Caracteres detectados", output)
    cv2.imshow("Resized", imgResize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
