# cloud_api.py – Minimaler Cloud-Stub für Snapshot → Session → Play
import os
import secrets
import time

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import sqlite3, time
from passlib.hash import bcrypt
import jwt  # PyJWT
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Basis-URL für die Web-Play-App (ENV überschreibbar)
PLAY_BASE_URL = os.getenv("PLAY_BASE_URL", "http://127.0.0.1:5080")

app = FastAPI(title="TripleOne Cloud Stub", version="0.0.1")


def init_auth_db():
    con = sqlite3.connect(AUTH_DB)
    con.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        created_at INTEGER NOT NULL
    );
    """
    )
    con.commit()
    con.close()


init_auth_db()


# In-Memory Stores (für Demo)
DEVICE_CONFIGS: dict[str, dict] = {}  # deviceId -> {"config": ..., "ts": ...}
TOKENS: dict[str, str] = {}  # one-time-token -> sessionId
SESSIONS: dict[str, dict] = {}  # sessionId -> {deviceId, created, options, ...}

AUTH_DB = os.path.join(os.path.dirname(__file__), "auth.db")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
bearer_scheme = HTTPBearer(auto_error=False)


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


@app.post("/auth/register")
async def auth_register(body: RegisterReq):
    if _user_by_email(body.email):
        raise HTTPException(400, "email already registered")
    if len(body.password) < 6:
        raise HTTPException(400, "password too short")
    _create_user(body.email, body.password, body.displayName)
    return {"ok": True}


@app.post("/auth/login")
async def auth_login(body: LoginReq):
    row = _user_by_email(body.email)
    if not row:
        raise HTTPException(401, "invalid credentials")
    uid, email, pw_hash, disp = row
    if not bcrypt.verify(body.password, pw_hash):
        raise HTTPException(401, "invalid credentials")
    token = _jwt_for_user(uid, email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": uid, "email": email, "displayName": disp},
    }


@app.get("/auth/me")
async def auth_me(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "unauthorized")
    return {"user": user}


class RegisterReq(BaseModel):
    email: str
    password: str
    displayName: str | None = None


class LoginReq(BaseModel):
    email: str
    password: str


def _user_by_email(email: str):
    con = sqlite3.connect(AUTH_DB)
    cur = con.execute(
        "SELECT id,email,password_hash,display_name FROM users WHERE email=?",
        (email.lower(),),
    )
    row = cur.fetchone()
    con.close()
    return row  # (id, email, hash, display_name) oder None


def _create_user(email: str, password: str, display_name: str | None):
    con = sqlite3.connect(AUTH_DB)
    con.execute(
        "INSERT INTO users(email,password_hash,display_name,created_at) VALUES(?,?,?,?)",
        (email.lower(), bcrypt.hash(password), display_name or "", int(time.time())),
    )
    con.commit()
    con.close()


def _jwt_for_user(uid: int, email: str) -> str:
    payload = {
        "sub": str(uid),
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 24 * 7,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _decode_token(token: str):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if creds is None or not creds.credentials:
        return None
    try:
        data = _decode_token(creds.credentials)
        return {"id": int(data["sub"]), "email": data["email"]}
    except Exception:
        return None


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5080)
