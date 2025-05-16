# app.py

import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
from PIL import Image
import requests
from bs4 import BeautifulSoup
import docx
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
from image_analyzer import analyze_image_sentiment
from sklearn.metrics import classification_report
import re
import requests
from bs4 import BeautifulSoup
from youtube_comment_downloader import YoutubeCommentDownloader

MODEL_PATH = "models/modernBERT"

@st.cache_resource
def load_model():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True)
    return tokenizer, model, config

tokenizer, model, config = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        label = config.id2label[pred_idx]
        confidence = probs[0][pred_idx].item()
    return label, confidence, probs[0]

def predict_sentiments_batch(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=1).tolist()
            labels = [config.id2label[idx] for idx in pred_ids]
        results.extend(labels)
    return results

def extract_text_from_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist())
    else:
        return ""

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["nav", "footer", "aside", "script", "style", "header"]):
            tag.decompose()
        article = soup.find("div", class_="article__text") or soup.find("article") or soup.find("main") or soup.body
        texts = article.stripped_strings
        result = " ".join(texts)
        return result[:3000]
    except Exception as e:
        return f"[Ошибка при загрузке: {e}]"

def plot_sentiment_distribution(probs):
    try:
        labels = list(config.id2label.values())
        values = probs.tolist()

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['green', 'orange', 'red'][:len(labels)])
        ax.set_ylabel("Вероятность")
        ax.set_title("Распределение тональности")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Ошибка при построении графика: {e}")


# Название приложения
st.markdown("<h1 class='main-title'>Мультиязычный анализ тональности текста</h1>", unsafe_allow_html=True)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
    html, body, .stApp {
        background-color: white !important;
        color: #222 !important;
        font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    }

    .block-container {
        padding: 2rem 2rem 4rem;
        background-color: white !important;
        max-width: 1000px;
        margin: auto;
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: white !important;
        visibility: hidden !important;
    }

    /* Заголовок сверху */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        color: #111;
    }

    /* Кнопки */
    .stButton button {
        background-color: #f8f9fa !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        color: #222 !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }

    .stButton button:hover {
        background-color: #e2e6ea !important;
        border-color: #999 !important;
        transform: scale(1.02);
    }

    /* Текстовые поля */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #fff !important;
        color: #222 !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        transition: border 0.2s ease-in-out;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #888 !important;
        outline: none !important;
    }

    /* Tabs */
    .stTabs [role="tab"] {
        background-color: #ffffff !important;
        color: #444 !important;
        font-weight: 500;
        border: 1px solid #ddd !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
        margin-right: 4px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #f5f5f5 !important;
        color: #000 !important;
        border-bottom: 2px solid black !important;
    }

    /* Код и результаты */
    code, pre, .stCodeBlock {
        background-color: #f5f5f5 !important;
        color: #222 !important;
        border-radius: 6px !important;
        padding: 6px;
        font-size: 14px;
    }

    /* Яркий вывод результата анализа */
    .result-box {
        background-color: #f8f8f8;
        color: #111 !important;
        padding: 1rem 1.5rem;
        border-left: 5px solid #ff4b4b;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* Таблицы */
    .stDataFrame {
        background-color: white !important;
        border: 1px solid #ccc !important;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }

    h1, h2, h3, h4, h5 {
        color: #111 !important;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Прокрутка */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #999;
    }

    ::-webkit-scrollbar-track {
        background: #f5f5f5;
    }

    /* Спиннер */
    .stSpinner > div {
        color: black !important;
    }

    /* Drag & Drop */
    [data-testid="stFileDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #111 !important;
        color: #111 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.write("Выберите источник текста для анализа:")
# Подключаем стили
st.markdown("""<style> /* Весь CSS, как у вас — тот же */ </style>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💬 Текст", "📁 Файл", "🖼️ Изображение", "🔗 URL", "📑 CSV-таблица",
    "🎥 YouTube комментарии", "📝 VK комментарии"
])

with tab1:
    user_input = st.text_area("Введите текст для анализа", height=150)
    if st.button("Анализировать текст", key="analyze_text"):
        if user_input.strip():
            label, confidence, probs = predict_sentiment(user_input)
            st.markdown(f"""
            <div class="result-box">
                📊 <strong>Результат анализа:</strong><br>
                <strong>Тональность:</strong> {label}<br>
                <strong>Уверенность:</strong> {confidence * 100:.1f}%
            </div>
            """, unsafe_allow_html=True)
            plot_sentiment_distribution(probs)
        else:
            st.warning("Пожалуйста, введите текст для анализа.")

with tab2:
    uploaded_file = st.file_uploader("Загрузите файл (txt, pdf, docx)", type=["txt", "pdf", "docx"])
    if uploaded_file:
        with st.spinner("📂 Извлечение текста..."):
            raw_text = extract_text_from_file(uploaded_file)  # ← сохраняем оригинал
        if raw_text:
            st.subheader("📄 Извлечённый текст (как в файле):")
            st.code(raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""))  # ← выводим без очистки

            label, confidence, probs = predict_sentiment(raw_text)  # ← анализ по исходному тексту

            st.markdown(f"""
            <div class="result-box">
                📊 <strong>Результат анализа:</strong><br>
                <strong>Тональность:</strong> {label}<br>
                <strong>Уверенность:</strong> {confidence * 100:.1f}%
            </div>
            """, unsafe_allow_html=True)

            plot_sentiment_distribution(probs)
        else:
            st.error("Не удалось извлечь текст из файла.")

with tab3:
    uploaded_image = st.file_uploader("Загрузите изображение (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        with st.spinner("🔍 Извлечение текста и анализ..."):
            result = analyze_image_sentiment(image, lambda text: {
                "label": predict_sentiment(text)[0],
                "confidence": predict_sentiment(text)[1]
            })
        st.subheader("📄 Извлечённый текст:")
        st.code(result.get("text") or "[нет текста]")
        if result.get("confidence") != "N/A":
            st.markdown(f"""
            <div class="result-box">
                📊 <strong>Результат анализа:</strong><br>
                <strong>Тональность:</strong> {result['label']}<br>
                <strong>Уверенность:</strong> {result['confidence'] * 100:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Не удалось распознать текст.")

with tab4:
    url = st.text_input("Введите ссылку на статью/страницу")
    if st.button("Анализировать ссылку"):
        if url:
            with st.spinner("🌐 Загрузка страницы и извлечение текста..."):
                text = extract_text_from_url(url)
            if text:
                st.subheader("📄 Извлечённый текст:")
                st.code(text[:1000] + ("..." if len(text) > 1000 else ""))
                label, confidence, probs = predict_sentiment(text)
                st.markdown(f"""
                <div class="result-box">
                    📊 <strong>Результат анализа:</strong><br>
                    <strong>Тональность:</strong> {label}<br>
                    <strong>Уверенность:</strong> {confidence * 100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                plot_sentiment_distribution(probs)
            else:
                st.error("Не удалось извлечь текст по ссылке.")

with tab5:
    st.header("📑 Анализ CSV-таблицы")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонками 'text' и 'label'", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")

            if "text" in df.columns and "label" in df.columns:
                label_map = {
                    "негативный": "negative", "нейтральный": "neutral", "позитивный": "positive",
                    "negative": "negative", "neutral": "neutral", "positive": "positive"
                }
                df["label"] = df["label"].map(label_map)

                if len(df) > 1000:
                    df = df.head(1000)
                    st.warning("⚠️ CSV слишком большой, анализируются только первые 1000 строк.")

                st.subheader("📄 Пример данных:")
                st.dataframe(df.head())

                with st.spinner("🔍 Анализируем..."):
                    texts = df["text"].astype(str).tolist()
                    y_true = df["label"].tolist()
                    y_pred = predict_sentiments_batch(texts)

                    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.subheader("📊 Классификационный отчёт:")
                    st.dataframe(report_df.round(3))

                    st.subheader("📈 Распределение предсказанных тональностей:")
                    pred_counts = pd.Series(y_pred).value_counts().reindex(config.id2label.values(), fill_value=0)
                    fig, ax = plt.subplots()
                    pred_counts.plot(kind='bar', color=['green', 'orange', 'red'][:len(pred_counts)], ax=ax)
                    ax.set_ylabel("Количество")
                    ax.set_title("Предсказанное распределение тональностей")
                    st.pyplot(fig)
            else:
                st.error("CSV-файл должен содержать колонки 'text' и 'label'.")
        except Exception as e:
            st.error(f"Ошибка при обработке CSV: {e}")
            

# ——————————————————————————————————————————————
# 1) Вспомогательные функции для YouTube и VK
# ——————————————————————————————————————————————

def extract_youtube_video_id(url: str) -> str:
    patterns = [r"(?:v=|\/)([0-9A-Za-z_-]{11})"]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

def get_youtube_comments(video_id, limit=50):
    from youtube_comment_downloader import YoutubeCommentDownloader

    try:
        limit = int(limit)
    except ValueError:
        return ["[Ошибка: limit должен быть числом]"]

    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        # Передаём sort_by=1 для сортировки по топу
        for comment in downloader.get_comments_from_url(
            f"https://www.youtube.com/watch?v={video_id}", sort_by=1
        ):
            text = comment.get("text", "")
            if isinstance(text, str) and text.strip():
                comments.append(text.strip())
                if len(comments) >= limit:
                    break
    except Exception as e:
        comments.append(f"[Ошибка YouTube: {e}]")
    return comments

def get_vk_post_comments(vk_url, limit=50):
    try:
        limit = int(limit)
    except ValueError:
        return ["[Ошибка: limit должен быть числом]"]
    resp = requests.get(vk_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    blocks = soup.find_all("div", class_="wall_reply_text")
    comments = [b.get_text(strip=True) for b in blocks]
    return comments[:limit] if comments else ["[Не найдено комментариев или пост закрыт]"]

# ——————————————————————————————————————————————
# 2) Вкладки YouTube (tab6) и VK (tab7)
# ——————————————————————————————————————————————

with tab6:
    st.header("🎥 Анализ комментариев YouTube")
    yt_url = st.text_input(
        "🔗 Ссылка на видео",
        placeholder="https://www.youtube.com/watch?v=... или https://youtu.be/..."
    )
    
    max_comments = st.slider(
        "Сколько комментариев загрузить", min_value=1, max_value=1000, value=100, step=10, key="yt_slider"
    )
    
    if st.button("📥 Загрузить и проанализировать", key="yt_button"):
        vid = extract_youtube_video_id(yt_url)
        if not vid:
            st.warning("Введите корректную ссылку на видео.")
        else:
            with st.spinner("🔍 Загружаем комментарии..."):
                comments = get_youtube_comments(vid, limit=max_comments)
            
            if comments:
                st.success(f"Загружено комментариев: {len(comments)}")
                st.dataframe(comments, use_container_width=True)
                
                with st.spinner("📊 Анализ тональности..."):
                    labels = predict_sentiments_batch(comments)
                    df = pd.DataFrame({"Комментарий": comments, "Тональность": labels})
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df["Тональность"].value_counts())
            else:
                st.warning("Не удалось получить комментарии.")


with tab7:
    st.header("📝 Анализ комментариев VK")
    vk_link = st.text_input("Введите ссылку на пост ВКонтакте")

    max_vk_comments = st.slider(
        "Сколько комментариев загрузить", min_value=1, max_value=1000, value=100, step=10
    )

    if st.button("Загрузить и проанализировать комментарии"):
        if vk_link:
            try:
                from vk_api_handler import fetch_vk_comments  # Убедитесь, что модуль доступен

                with st.spinner("📥 Получаем комментарии..."):
                    comments = fetch_vk_comments(vk_link, max_comments=max_vk_comments)

                if comments:
                    st.success(f"🔍 Получено комментариев: {len(comments)}")
                    st.subheader("📄 Пример комментариев:")
                    st.write(comments[:10])

                    with st.spinner("🤖 Анализируем тональность..."):
                        preds = predict_sentiments_batch(comments)
                        df = pd.DataFrame({"Комментарий": comments, "Тональность": preds})

                        st.subheader("📊 Результаты анализа:")
                        st.dataframe(df)

                        # Гистограмма
                        st.subheader("📈 Распределение тональности:")
                        sentiment_counts = df["Тональность"].value_counts()
                        fig, ax = plt.subplots()
                        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'orange', 'green'])
                        ax.set_xlabel("Тональность")
                        ax.set_ylabel("Количество")
                        st.pyplot(fig)
                else:
                    st.warning("Не удалось получить комментарии.")
            except Exception as e:
                st.error(f"Ошибка при получении или анализе: {e}")
        else:
            st.warning("Пожалуйста, введите ссылку на пост ВКонтакте.")
