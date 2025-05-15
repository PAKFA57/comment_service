# modernbert_predict.py

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch

# Путь к локальной папке с моделью
MODEL_PATH = "models/modernBERT"

# Загрузка конфигурации
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Загрузка токенизатора и модели из локальной папки
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        label = config.id2label[pred_idx]
        confidence = probs[0][pred_idx].item()
    return label, round(confidence, 3)

if __name__ == "__main__":
    print("Введите текст для анализа (или 'exit' для выхода):")
    while True:
        text = input(">> ")
        if text.strip().lower() == "exit":
            break
        label, conf = predict_sentiment(text)
        print(f"Тональность: {label} ({conf * 100:.1f}% уверенности)")
