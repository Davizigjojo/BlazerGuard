import os
import gc
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from tokenizers import Tokenizer
import onnxruntime as ort

app = FastAPI()

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
    print("AVISO: tokenizer.json nao encontrado!")

print("2. Carregando Sessao ONNX Compacta (72.5MB)...")
if os.path.exists(MODEL_FILE):
    onnx_options = ort.SessionOptions()
    onnx_options.intra_op_num_threads = 1
    onnx_options.inter_op_num_threads = 1
    onnx_options.enable_cpu_mem_arena = False  # Impede estouro de RAM no Render
    onnx_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    ort_session = ort.InferenceSession(MODEL_FILE, onnx_options)
    print("Sessao Guardrail Llama ativa e operacional!")
else:
    ort_session = None
    print("AVISO: model.quant.onnx nao encontrado!")

gc.collect()

class ContentCheckRequest(BaseModel):
    text: str

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.get("/")
async def root():
    # ROTA DE PING ISOLADA: Retorna sucesso instantaneo para o UptimeRobot
    # Sem tocar na memoria do ONNX Runtime, evitando falhas de inicializacao
    status_modelo = "Pronto" if ort_session is not None else "Aguardando Arquivos"
    return {
        "status": "online", 
        "message": "BlazerGuard Llama 22M operacional.",
        "model_status": status_modelo
    }

@app.post("/check")
async def check_content(request: ContentCheckRequest, x_custom_auth_token: str = Header(None)):
    if x_custom_auth_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Não autorizado")

    if ort_session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modelo Guardrail temporariamente indisponivel.")

    try:
        encoding = tokenizer.encode(request.text)
        
        # Cria os tensores no formato Int64 nativo
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # Executa a classificacao de seguranca
        outputs = ort_session.run(None, onnx_inputs)
        logits = np.array(outputs[0])
        
        # Converte a matriz de logits usando sigmoid seguro para arrays
        probabilities = sigmoid(logits)
        
        # Tratamento seguro para extrair o score maximo independente do formato do array
        score_bruto = np.max(probabilities)
        injection_score = float(score_bruto)

        # Se o modelo der mais de 65% de chance de ser uma burla/ataque, barra
        IS_UNSAFE = bool(injection_score > 0.65)

        return {
            "is_unsafe": IS_UNSAFE,
            "score": injection_score
        }
        
    except Exception as erro_interno:
        # Evita derrubar o servidor HTTP caso o formato do texto de entrada bugue
        raise HTTPException(status_code=500, detail=f"Erro no processamento da IA: {str(erro_interno)}")

if __name__ == "__main__":
    import uvicorn
    # Converte dinamicamente a porta para evitar o erro 502 de mapeamento do Render
    port_env = os.environ.get("PORT", "10000").strip()
    port = int(port_env) if port_env.isdigit() else 10000
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    
