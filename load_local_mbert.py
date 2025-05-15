# load_local_mbert.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path: str = "models/modernBERT"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model
