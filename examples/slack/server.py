import os

import uvicorn

# Import our Slack agent to handle webhook events
from agent import agent
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pyngrok import ngrok

load_dotenv()

app = FastAPI()

@app.post("/")
async def slack_events(request: Request):
    body = await request.json()
    
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}
    
    # Process actual Slack events here
    if body.get("type") == "event_callback":
        await agent(_webhook=body).collect()  # _webhook is a custom parameter name you define
        
    return {"status": "ok"}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "4343"))
    except Exception:
        port = 4343
    public_url = ngrok.connect(port, "http")
    uvicorn.run(app, host=host, port=port)