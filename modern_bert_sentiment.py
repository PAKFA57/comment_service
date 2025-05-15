# modern_bert_sentiment.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import argparse

MODEL_NAME = "clapAI/modernBERT-base-multilingual-sentiment"

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        label = config.id2label[pred_idx]
        confidence = probs[0][pred_idx].item()
    return label, round(confidence, 3)

def interactive_mode():
    print("🔍 Введите текст (или 'exit' для выхода):")
    while True:
        text = input(">> ")
        if text.strip().lower() == "exit":
            break
        label, conf = predict_sentiment(text)
        print(f"Предсказание: {label} ({conf * 100:.1f}% уверенности)")

def train_on_custom_data(csv_path):
    # Ожидается CSV с колонками: text, label
    df = pd.read_csv(csv_path)
    label2id = {label: i for i, label in enumerate(config.id2label.values())}
    df = df[df["label"].isin(label2id.keys())]  # фильтрация допустимых меток

    dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = [label2id[label] for label in batch["label"]]
        return tokens

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./modern_bert_finetuned",
        evaluation_strategy="no",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("./modern_bert_finetuned")
    tokenizer.save_pretrained("./modern_bert_finetuned")
    print("✅ Дообучение завершено. Модель сохранена в './modern_bert_finetuned'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", help="Путь к CSV для дообучения")
    args = parser.parse_args()

    if args.train_csv:
        train_on_custom_data(args.train_csv)
    else:
        interactive_mode()
