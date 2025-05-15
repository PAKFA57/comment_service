# analyze_model.py

import torch
import torch.nn.functional as F
from load_local_mbert import load_model
from preprocess import preprocess_text

def predict_sentiment(text: str):
    tokenizer, model = load_model()
    model.eval()

    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
    return pred, probs.squeeze().tolist()
