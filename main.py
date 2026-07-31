import os
import asyncio
import websockets
import json
import requests

# URL da tua IA na Hugging Face
IA_URL = "https://davizig10jojo-ia.hf.space/perguntar_mc"

async def handler(websocket):
    print("Cliente (Minecraft) conectado ao Bridge!")
    try:
        async for message in websocket:
            dados = json.loads(message)
            pergunta = dados.get("texto", "")
            
            # Envia para a Hugging Face
            try:
                response = requests.post(IA_URL, json={"texto": pergunta})
                if response.status_code == 200:
                    resposta_ia = response.json().get("resposta", "Sem resposta da IA.")
                else:
                    resposta_ia = "Erro temporário no servidor da IA."
            except Exception:
                resposta_ia = "Erro de conexão com o modelo de IA."
            
            # Envia a resposta de volta para o Minecraft
            await websocket.send(json.dumps({"resposta": resposta_ia}))
            
    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")

async def main():
    # A Render fornece a porta automaticamente pela variável de ambiente PORT
    port = int(os.environ.get("PORT", 8080))
    # '0.0.0.0' escuta em todas as interfaces públicas da nuvem
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"Bridge a correr na porta {port}...")
        await asyncio.Future()  # Mantém o servidor ligado continuamente

if __name__ == "__main__":
    asyncio.run(main())
