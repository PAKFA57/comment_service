import os
import requests
import pytesseract
import tempfile
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from PIL import Image
from io import BytesIO
import textract
import pandas as pd

# Для стабильного результата в langdetect
DetectorFactory.seed = 0

SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx']

# 💬 Определение языка текста
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    return lang

# 📤 Извлечение текста в зависимости от типа данных
def extract_text(input_type, input_data, lang_hint=None):
    """
    Универсальная функция для извлечения текста из разных источников.
    
    :param input_type: тип входных данных: 'text', 'file', 'url', 'image'
    :param input_data: данные, которые нужно обработать (строка, файл, URL или изображение)
    :param lang_hint: подсказка для языка текста (если это изображение)
    :return: извлечённый текст или сообщение об ошибке
    """
    if input_type == "text":
        return input_data
    elif input_type == "file":
        return load_text_from_file(input_data)
    elif input_type == "url":
        return load_text_from_url(input_data)
    elif input_type == "image":
        lang_code = lang_hint or "eng"
        return load_text_from_image(input_data, lang=lang_code)
    else:
        return "❌ Unknown input type"

# 📤 Загрузка текста из файла
def load_text_from_file(uploaded_file):
    """
    Загрузка текста из поддерживаемых файлов (.txt, .pdf, .docx).
    """
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if file_ext == ".txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif file_ext in [".pdf", ".docx"]:
            return textract.process(tmp_path).decode("utf-8", errors="ignore")
        else:
            return f"❌ Unsupported file extension: {file_ext}"
    except Exception as e:
        return f"❌ Error extracting file text: {e}"
    finally:
        os.remove(tmp_path)

# 🌐 Загрузка текста с URL
def load_text_from_url(url):
    """
    Загрузка текста с указанного URL.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Удаляем скрипты, стили и нежелательный HTML
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        # Получаем чистый текст
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error fetching URL content: {e}"

# 🖼️ Извлечение текста из изображения
def load_text_from_image(uploaded_image, lang="eng"):
    """
    Извлечение текста из изображения с использованием OCR.
    """
    try:
        image = Image.open(uploaded_image).convert("RGB")
        # Преобразуем в оттенки серого и применяем OCR
        gray = image.convert("L")
        text = pytesseract.image_to_string(gray, lang=lang)
        return text.strip()
    except Exception as e:
        return f"❌ Error processing image: {e}"

