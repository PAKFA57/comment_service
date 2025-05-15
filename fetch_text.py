import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re

def fetch_text_from_file(file):
    try:
        if file.name.endswith(".txt"):
            return file.read().decode("utf-8").splitlines()
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            if "text" in df.columns:
                return df["text"].tolist()
    except Exception as e:
        print(f"Ошибка обработки файла: {e}")
    return []

def fetch_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Ошибка HTTP: {response.status_code}")
            return None
        
        article = Article(url, language='ru')
        article.download(input_html=response.text) 
        article.parse()
        
        text = article.text.strip()
        if not text or len(text.split()) < 50:
            return extract_text_using_bs4(response.text)
        
        # Очистка текста
        text = clean_article_text(text)
        return text
    except Exception as e:
        print(f"Ошибка извлечения текста: {e}")
        return None

def extract_text_using_bs4(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    return text.strip()

def clean_article_text(text):
    text = re.sub(r'Фото:.*?(?=\.\s|$)', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()
