import os
import gc
import numpy as np

# Configurações de ambiente para reduzir o footprint de memória RAM do Hugging Face
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import onnxruntime as ort

app = FastAPI()

# Senha do BlazerGuard
SECRET_TOKEN = "BlazerGuard_MinhaSenhaSecreta123xpto"

MODEL_REPO = "gravitee-io/Llama-Prompt-Guard-2-86M-onnx"
MODEL_FILE = "model.quant.onnx"

# Download otimizado do modelo direto do HF Hub para economizar RAM
try:
    print("Baixando/Verificando o arquivo .onnx via huggingface_hub...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Modelo carregado no caminho: {model_path}")
except Exception as e:
    print(f"Erro ao baixar modelo do HF Hub: {e}")
    model_path = None

# Configurações estritas do ONNX Runtime para Render Free (1 vCPU, 512MB RAM)
onnx_options = ort.SessionOptions()
onnx_options.intra_op_num_threads = 1
onnx_options.inter_op_num_threads = 1
onnx_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

print("Carregando Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

if model_path and os.path.exists(model_path):
    print("Carregando Sessão ONNX Runtime...")
    ort_session = ort.InferenceSession(model_path, onnx_options)
    print("Sessão ONNX iniciada com sucesso!")
else:
    ort_session = None
    print("AVISO: Modelo ONNX não pôde ser carregado.")

# Força a liberação de memória acumulada durante o boot
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

    if ort_session is None:
        raise HTTPException(status_code=503, detail="Modelo de proteção do BlazerIA não carregado.")

    inputs = tokenizer(request.text, return_tensors="np", max_length=128, truncation=True)
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
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
    
