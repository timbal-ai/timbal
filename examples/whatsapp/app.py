from fastapi import FastAPI, Request, Response
from agent import agent
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

@app.get("/")
async def verify(request: Request):
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if token == VERIFY_TOKEN and challenge:
        return Response(content=challenge, media_type="text/plain")
    return Response(content="Verification failed", status_code=403)

@app.post("/")
async def webhook(request: Request):
    _webhook = await request.json()
    await agent(_webhook=_webhook).collect()
    return {"status": "ok"}

def _maybe_start_ngrok(port: int) -> None:
    enable = os.getenv("ENABLE_NGORK", "false")
    if enable != "true":
        return
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(port, "http")
        print(f"ngrok public url: {public_url}")
    except Exception as e:
        print(f"Failed to start ngrok: {e}")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "4343"))
    except Exception:
        port = 4343
    _maybe_start_ngrok(port)
    uvicorn.run("app:app", host=host, port=port, log_level="info")
    