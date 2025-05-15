# train_model.py

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import evaluate
import os

# === Конфигурация ===
MODEL_NAME = "clapAI/modernBERT-base-multilingual-sentiment"
DATA_PATH = "datasets/my_sentiment_data.csv"
OUTPUT_DIR = "models/modernBERT"
NUM_EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5

# === Загрузка данных ===
df = pd.read_csv(DATA_PATH)
label_list = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["label"].map(label2id)
dataset = Dataset.from_pandas(df)

# === Токенизация ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(tokenize)

# === Загрузка модели ===
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True
)

# === Метрика ===
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# === Параметры обучения ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset.shuffle(seed=42).select(range(len(encoded_dataset)//5)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === Обучение ===
trainer.train()

# === Сохранение модели ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
