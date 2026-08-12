import os
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

app = FastAPI()

# Senha do BlazerGuard
SECRET_TOKEN = "BlazerGuard_MinhaSenhaSecreta123xpto"

MODEL_REPO = "gravitee-io/Llama-Prompt-Guard-2-86M-onnx"
MODEL_FILE = "model.quant.onnx"

# URL CORRIGIDA: Adicionada a barra '/' após .co
URL_MODELO = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    print("Iniciando download do modelo via Requests...")
    try:
        response = requests.get(URL_MODELO, stream=True, timeout=120)
        response.raise_for_status()
        with open(MODEL_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download concluído com sucesso!")
    except Exception as e:
        print(f"Erro ao baixar o modelo: {e}")

# Configurações para otimização em instâncias com recursos limitados (Render)
onnx_options = ort.SessionOptions()
onnx_options.intra_op_num_threads = 1
onnx_options.inter_op_num_threads = 1
onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL 

print("Carregando o Tokenizer e a sessão ONNX...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

if os.path.exists(MODEL_FILE):
    ort_session = ort.InferenceSession(MODEL_FILE, onnx_options)
    print("Sessão do ONNX iniciada com sucesso!")
else:
    ort_session = None
    print("AVISO: O arquivo .onnx não foi encontrado.")

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

# Bloco de execução com suporte à porta dinâmica do Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
