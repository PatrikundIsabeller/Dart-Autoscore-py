# cloud_api.py – Minimaler Cloud-Stub für Snapshot → Session → Play
import os
import secrets
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Basis-URL für die Web-Play-App (ENV überschreibbar)
PLAY_BASE_URL = os.getenv("PLAY_BASE_URL", "http://127.0.0.1:5080")

app = FastAPI(title="TripleOne Cloud Stub", version="0.0.1")

# In-Memory Stores (für Demo)
DEVICE_CONFIGS: dict[str, dict] = {}  # deviceId -> {"config": ..., "ts": ...}
TOKENS: dict[str, str] = {}  # one-time-token -> sessionId
SESSIONS: dict[str, dict] = {}  # sessionId -> {deviceId, created, options, ...}


class SessionReq(BaseModel):
    deviceId: str
    gameType: str = "X01"
    options: dict | None = None


@app.post("/v1/devices/{deviceId}/config/snapshot")
async def snapshot(deviceId: str, req: Request):
    cfg = await req.json()
    if not isinstance(cfg, dict):
        raise HTTPException(400, "invalid config")
    DEVICE_CONFIGS[deviceId] = {"config": cfg, "ts": time.time()}
    return {"ok": True, "deviceId": deviceId, "ts": DEVICE_CONFIGS[deviceId]["ts"]}


@app.post("/v1/sessions")
async def create_session(body: SessionReq):
    if body.deviceId not in DEVICE_CONFIGS:
        raise HTTPException(400, "no config snapshot for device")
    sid = f"ses_{secrets.token_urlsafe(8)}"
    token = f"ptk_{secrets.token_urlsafe(24)}"  # One-Time-Token
    SESSIONS[sid] = {
        "sessionId": sid,
        "deviceId": body.deviceId,
        "created": time.time(),
        "options": body.options or {},
        "gameType": body.gameType,
        "configTs": DEVICE_CONFIGS[body.deviceId]["ts"],
    }
    TOKENS[token] = sid
    return {
        "sessionId": sid,
        "oneTimeToken": token,
        "playUrl": f"{PLAY_BASE_URL}/play",  # Next.js-Seite /s?token=...
    }


@app.post("/v1/sessions/redeem")
async def redeem(body: dict):
    token = body.get("token")
    if not token or token not in TOKENS:
        raise HTTPException(401, "invalid/expired token")
    sid = TOKENS.pop(token)  # single-use
    ses = SESSIONS[sid]
    cfg = DEVICE_CONFIGS[ses["deviceId"]]["config"]
    return {
        "sessionId": sid,
        "deviceId": ses["deviceId"],
        "config": cfg,
        "options": ses["options"],
        "gameType": ses["gameType"],
    }


# >>> NEU: Session-Config abrufen (für die Play-Seite)
@app.get("/v1/sessions/{sid}/config")
async def session_config(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "session not found")
    ses = SESSIONS[sid]
    if ses["deviceId"] not in DEVICE_CONFIGS:
        raise HTTPException(404, "config not found")
    cfg = DEVICE_CONFIGS[ses["deviceId"]]["config"]
    return {
        "sessionId": sid,
        "deviceId": ses["deviceId"],
        "config": cfg,
        "options": ses["options"],
        "gameType": ses["gameType"],
    }


# Optionaler HTML-Stub (nicht benötigt, wenn Next.js /s verwendet wird)
@app.get("/play", response_class=HTMLResponse)
async def play(token: str | None = None):
    if not token or token not in TOKENS:
        return HTMLResponse(
            "<h1>Token fehlt oder ist abgelaufen.</h1>", status_code=401
        )
    sid = TOKENS.pop(token)
    ses = SESSIONS[sid]
    cfg = DEVICE_CONFIGS[ses["deviceId"]]["config"]
    html = f"""
    <html>
    <head><title>TripleOne Play (Stub)</title></head>
    <body style="font-family:Segoe UI,Arial;padding:24px">
      <h1>Session {sid}</h1>
      <p>Device: <b>{ses['deviceId']}</b></p>
      <p>Game: <b>{ses['gameType']}</b></p>
      <h3>Config Snapshot</h3>
      <pre style="background:#111;color:#0f0;padding:12px;border-radius:8px;max-width:900px;overflow:auto;">{cfg}</pre>
      <p>✔ Token eingelöst. Hier würdest du jetzt die echte Spiel-UI laden.</p>
    </body>
    </html>
    """
    return HTMLResponse(html)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5080)
