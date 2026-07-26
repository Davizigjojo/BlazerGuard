from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

MODEL_REPO = "Davizig10jojo/BlazerGuard-pt-BR"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "API de Moderação Online"}

@app.post("/predict")
def predict(data: TextRequest):
    if not data.text.strip():
        return {"is_toxic": False}
        
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        
    return {"is_toxic": bool(pred == 1)}
