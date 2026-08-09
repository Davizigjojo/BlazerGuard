import os
import urllib.request
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

app = FastAPI()

# Sua senha inserida diretamente no código para facilitar o deploy no Render
SECRET_TOKEN = "BlazerGuard_MinhaSenhaSecreta123xpto"

MODEL_REPO = "gravitee-io/Llama-Prompt-Guard-2-86M-onnx"
MODEL_FILE = "model.quant.onnx"

# Baixa o modelo direto do Hugging Face para o servidor do Render automaticamente
if not os.path.exists(MODEL_FILE):
    print("Baixando o arquivo ONNX quantizado...")
    url = f"https://huggingface.co{MODEL_REPO}/resolve/main/{MODEL_FILE}"
    urllib.request.urlretrieve(url, MODEL_FILE)

# Configurações para travar o consumo de memória e CPU nos limites do plano gratuito
onnx_options = ort.SessionOptions()
onnx_options.intra_op_num_threads = 1
onnx_options.inter_op_num_threads = 1
# Linha essencial: impede o ONNX de quebrar o contêiner do Render
onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL 

print("Carregando o Tokenizer e a sessão ONNX...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
ort_session = ort.InferenceSession(MODEL_FILE, onnx_options)

class ContentCheckRequest(BaseModel):
    text: str

@app.post("/check")
async def check_content(request: ContentCheckRequest, x_custom_auth_token: str = Header(None)):
    # Validação estrita da sua senha do BlazerGuard
    if x_custom_auth_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Não autorizado")

    # Limita o tamanho do texto para economizar processamento e memória RAM
    inputs = tokenizer(request.text, return_tensors="np", max_length=128, truncation=True)
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Roda a inteligência de segurança
    outputs = ort_session.run(None, onnx_inputs)
    logits = outputs
    
    # Transforma o resultado em uma probabilidade de 0.0 a 1.0
    probability = 1 / (1 + np.exp(-logits)) 
    
    # Se passar de 60% de certeza que é um ataque ou comando malicioso, bloqueia
    IS_UNSAFE = bool(probability > 0.60)

    return {
        "is_unsafe": IS_UNSAFE,
        "score": float(probability)
    }
