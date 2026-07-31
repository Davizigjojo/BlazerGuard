import os
import asyncio
import websockets
import json
import requests

IA_URL = "https://davizig10jojo-ia.hf.space/perguntar_mc"

async def handler(websocket):
    print("Minecraft conectado com sucesso ao Bridge WebSocket!")
    try:
        async for message in websocket:
            print(f"Mensagem recebida do jogo: {message}")
            pergunta = message
            
            # Tenta extrair JSON caso o Minecraft envie estruturado
            try:
                dados = json.loads(message)
                if isinstance(dados, dict):
                    pergunta = dados.get("body", {}).get("message", dados.get("texto", message))
            except:
                pass
            
            # Envia para a IA da Hugging Face
            try:
                response = requests.post(IA_URL, json={"texto": str(pergunta)}, timeout=15)
                if response.status_code == 200:
                    resposta_ia = response.json().get("resposta", "Sem resposta da IA.")
                else:
                    resposta_ia = "Erro temporário no servidor da IA."
            except Exception as e:
                print(f"Erro ao contactar a IA: {e}")
                resposta_ia = "Erro de conexão com o modelo de IA."
            
            # Resposta formatada de volta para o jogo
            resposta_json = json.dumps({
                "body": {
                    "statusMessage": f"§5[BlazerIA] §f{resposta_ia}"
                },
                "header": {
                    "requestId": "00000000-0000-0000-0000-000000000000",
                    "messagePurpose": "commandResponse",
                    "version": 1,
                    "messageType": "commandResponse"
                }
            })
            
            await websocket.send(resposta_json)
            
    except websockets.exceptions.ConnectionClosed:
        print("Conexão WebSocket fechada pelo cliente.")

async def main():
    port = int(os.environ.get("PORT", 10000))
    # '0.0.0.0' permite conexões externas públicas
    async with websockets.serve(handler, "0.0.0.0", port, ping_interval=None):
        print(f"Servidor WebSocket ativo e a escuta na porta {port}...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
