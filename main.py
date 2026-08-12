import os
import gc
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

app = FastAPI()

SECRET_TOKEN = "BlazerGuard_MinhaSenhaSecreta123xpto"
MODEL_REPO = "gravitee-io/Llama-Prompt-Guard-2-86M-onnx"
MODEL_FILE = "model.quant.onnx"

print("1. Baixando arquivos do modelo e do tokenizer...")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    tokenizer_path = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer.json")
    print("Arquivos obtidos com sucesso!")
except Exception as e:
    print(f"Erro ao baixar arquivos do HF Hub: {e}")
    model_path = None
    tokenizer_path = None

print("2. Carregando Tokenizer leve...")
if tokenizer_path and os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # Habilita truncamento automático para não exceder 128 tokens
    tokenizer.enable_truncation(max_length=128)
else:
    tokenizer = None

print("3. Carregando Sessão ONNX Runtime...")
if model_path and os.path.exists(model_path):
    onnx_options = ort.SessionOptions()
    onnx_options.intra_op_num_threads = 1
    onnx_options.inter_op_num_threads = 1
    onnx_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    ort_session = ort.InferenceSession(model_path, onnx_options)
    print("Sessão ONNX pronta!")
else:
    ort_session = None

# Força a liberação do lixo de memória na inicialização
gc.collect()

class ContentCheckRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"status": "online", "message": "BlazerGuard ativo e operacional."}

@app.post("/check")
async def check_content(request: ContentCheckRequest, x_custom_auth_token: str = Header(None)):
    if x_custom_auth_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Não autorizado")

    if ort_session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modelo de proteção do BlazerIA não carregado.")

    # Tokenização leve e rápida
    encoding = tokenizer.encode(request.text)
    
    onnx_inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64)
    }

    outputs = ort_session.run(None, onnx_inputs)
    logits = outputs[0]
    
    probability = 1 / (1 + np.exp(-np.array(logits))) 
    score = float(np.max(probability))

    return {
        "is_unsafe": bool(score > 0.60),
        "score": score
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    
