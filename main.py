from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_REPO = "Davizig10jojo/BlazerGuard-pt-BR"

print("Carregando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

print("Baixando e carregando modelo ONNX...")
# Baixa o arquivo model.onnx direto do seu repositório HF
onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename="onnx/model.onnx")

# Inicializa o ONNX Runtime (consome pouquíssima RAM)
session = ort.InferenceSession(onnx_path)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "API de Moderação ONNX Online"}

@app.post("/predict")
def predict(data: TextRequest):
    if not data.text.strip():
        return {"level": "safe", "score": 0.0}

    # Tokenização
    inputs = tokenizer(data.text, return_tensors="np", padding=True, truncation=True, max_length=128)
    
    onnx_inputs = {
        session.get_inputs()[0].name: inputs["input_ids"].astype(np.int64),
        session.get_inputs()[1].name: inputs["attention_mask"].astype(np.int64)
    }

    # Inferência leve
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]

    # Transforma logits em probabilidades (Softmax)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    toxic_score = float(probs[0][1])

    # Categorização de gravidade
    if toxic_score >= 0.85:
        level = "unsafe"
    elif toxic_score >= 0.50:
        level = "controversial"
    else:
        level = "safe"

    return {"level": level, "score": round(toxic_score, 4)}
