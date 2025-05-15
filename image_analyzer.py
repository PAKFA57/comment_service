# image_analyzer.py

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\danil\Desktop\NIRkon\ts\tesseract.exe"

import cv2
from PIL import Image
import numpy as np
import tempfile
import os
from typing import Callable, Dict, Union


def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Не удалось загрузить изображение.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Увеличиваем изображение
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Повышаем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Размытие
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Порог + инверсия (если текст светлый)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = 255 - binary  # инвертируем изображение

    # Морфологическая операция — расширение символов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.dilate(inverted, kernel, iterations=1)

    return morph



def extract_text_from_image(image_file: Union[Image.Image, str], lang: str = "rus+eng") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp_path = tmp.name
        if isinstance(image_file, Image.Image):
            image_file.save(tmp_path)
        else:
            img = Image.open(image_file)
            img.save(tmp_path)

    try:
        processed_img = preprocess_image(tmp_path)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed_img, lang=lang, config=config)
        return text.strip()
    finally:
        os.remove(tmp_path)


def analyze_image_sentiment(image_file: Union[Image.Image, str], analyze_fn: Callable[[str], Dict[str, float]]) -> Dict[str, Union[str, float]]:
    text = extract_text_from_image(image_file)
    if not text:
        return {
            "text": "",
            "label": "Не удалось извлечь текст",
            "confidence": "N/A"
        }

    result = analyze_fn(text)
    return {
        "text": text,
        "label": result["label"],
        "confidence": result["confidence"]
    }
