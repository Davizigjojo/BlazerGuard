import json
import re
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_REPO = "Davizig10jojo/BlazerGuard-pt-BR"

print("Baixando modelo ONNX e configurações do Tokenizer...")
onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename="onnx/model.onnx")
vocab_path = hf_hub_download(repo_id=MODEL_REPO, filename="vocab.txt")
tokenizer_config_path = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer_config.json")

# Carrega o vocabulário em memória (apenas um dicionário leve em Python)
vocab = {}
with open(vocab_path, "r", encoding="utf-8") as f:
    for index, line in enumerate(f):
        vocab[line.strip()] = index

# Configura as opções do ONNX Runtime para economizar memória máxima
session_options = ort.SessionOptions()
session_options.enable_cpu_mem_arena = False
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])

def basic_tokenizer(text: str, max_length: int = 128):
    """
    Tokenizer minimalista estilo WordPiece (consome 0 MB adicionais de RAM).
    """
    text = text.lower().strip()
    tokens = ["[CLS]"]
    
    # Separa palavras e pontuações
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    
    for word in words:
        if word in vocab:
            tokens.append(word)
        else:
            # Subpalavras (WordPiece básico)
            i = 0
            while i < len(word):
                j = len(word)
                cur_substr = None
                while j > i:
                    substr = word[i:j]
                    if i > 0:
                        substr = "##" + substr
                    if substr in vocab:
                        cur_substr = substr
                        break
                    j -= 1
                if cur_substr is None:
                    tokens.append("[UNK]")
                    break
                tokens.append(cur_substr)
                i = j

    tokens = tokens[:max_length - 1] + ["[SEP]"]
    
    input_ids = [vocab.get(token, vocab.get("[UNK]", 100)) for token in tokens]
    attention_mask = [1] * len(input_ids)
    
    # Preenchimento (Padding)
    padding_length = max_length - len(input_ids)
    input_ids += [vocab.get("[PAD]", 0)] * padding_length
    attention_mask += [0] * padding_length
    
    return np.array([input_ids], dtype=np.int64), np.array([attention_mask], dtype=np.int64)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "API de Moderação ONNX Ultra-Leve Online"}

@app.post("/predict")
def predict(data: TextRequest):
    if not data.text.strip():
        return {"level": "safe", "score": 0.0}

    input_ids, attention_mask = basic_tokenizer(data.text)
    
    onnx_inputs = {
        session.get_inputs()[0].name: input_ids,
        session.get_inputs()[1].name: attention_mask
    }

    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]

    # Softmax leve
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    toxic_score = float(probs[0][1])

    if toxic_score >= 0.85:
        level = "unsafe"
    elif toxic_score >= 0.50:
        level = "controversial"
    else:
        level = "safe"

    return {"level": level, "score": round(toxic_score, 4)}
