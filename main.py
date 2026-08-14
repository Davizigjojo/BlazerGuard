import os
import gc
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from tokenizers import Tokenizer
import onnxruntime as ort

app = FastAPI()

# Sua senha de comunicação com o Cloudflare Worker
SECRET_TOKEN = "BlazerGuard_MinhaSenhaSecreta123xpto"
MODEL_FILE = "model.quant.onnx"
TOKENIZER_FILE = "tokenizer.json"

print("1. Carregando Tokenizer do Llama Guard 22M...")
if os.path.exists(TOKENIZER_FILE):
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    tokenizer.enable_truncation(max_length=256)
    print("Tokenizer carregado com sucesso!")
else:
    tokenizer = None
    print("ERRO CRÍTICO: tokenizer.json nao encontrado!")

print("2. Carregando Sessao ONNX Compacta (72.5MB)...")
if os.path.exists(MODEL_FILE):
    onnx_options = ort.SessionOptions()
    onnx_options.intra_op_num_threads = 1
    onnx_options.inter_op_num_threads = 1
    onnx_options.enable_cpu_mem_arena = False  # Impede o estouro de RAM no Render
    onnx_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    ort_session = ort.InferenceSession(MODEL_FILE, onnx_options)
    print("Sessao Guardrail Llama ativa e operacional!")
else:
    ort_session = None
    print("ERRO CRÍTICO: model.quant.onnx nao encontrado!")

# Força a limpeza da memória RAM após carregar a IA
gc.collect()

class ContentCheckRequest(BaseModel):
    text: str

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.get("/")
async def root():
    return {"status": "online", "message": "BlazerGuard Llama 22M operacional."}

@app.post("/check")
async def check_content(request: ContentCheckRequest, x_custom_auth_token: str = Header(None)):
    if x_custom_auth_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Não autorizado")

    if ort_session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modelo Guardrail temporariamente indisponivel.")

    encoding = tokenizer.encode(request.text)
    
    # Prepara a entrada exata em tensores int64 esperada pelo Llama Guard
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    
    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # Executa a classificação de segurança
    outputs = ort_session.run(None, onnx_inputs)
    logits = outputs[0]
    
    # Converte os resultados brutos em probabilidades de 0.0 a 1.0
    probabilities = softmax(logits)[0]
    
    # Na arquitetura Llama Guard binária, o índice 1 mapeia o comportamento inadequado (Anomalia/Ataque)
    injection_score = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

    # Se o modelo der mais de 65% de certeza de que é um ataque em português, barra
    IS_UNSAFE = bool(injection_score > 0.65)

    return {
        "is_unsafe": IS_UNSAFE,
        "score": injection_score
    }

if __name__ == "__main__":
    import uvicorn
    # Converte explicitamente a porta para inteiro e limpa espaços vazios
    port_env = os.environ.get("PORT", "10000").strip()
    port = int(port_env) if port_env.isdigit() else 10000
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    
