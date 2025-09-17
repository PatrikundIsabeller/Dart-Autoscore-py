# auth_client.py
import os, json, requests
from pathlib import Path

APPDIR = Path(os.getenv("APPDATA", ".")) / "TripleOne"
APPDIR.mkdir(parents=True, exist_ok=True)
CFG_FILE = APPDIR / "config.json"


class AuthClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token: str | None = None
        self.load_token()

    def headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def save_token(self, token: str | None):
        self.token = token
        data = {}
        if CFG_FILE.exists():
            try:
                data = json.loads(CFG_FILE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data.setdefault("auth", {})["token"] = token
        CFG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_token(self):
        if CFG_FILE.exists():
            try:
                data = json.loads(CFG_FILE.read_text(encoding="utf-8"))
                self.token = data.get("auth", {}).get("token")
            except Exception:
                self.token = None

    # --- API calls ---
    def register(self, email: str, password: str, display_name: str | None):
        r = requests.post(
            self.base_url + "/auth/register",
            json={"email": email, "password": password, "displayName": display_name},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()

    def login(self, email: str, password: str):
        r = requests.post(
            self.base_url + "/auth/login",
            json={"email": email, "password": password},
            timeout=5,
        )
        r.raise_for_status()
        js = r.json()
        self.save_token(js.get("access_token"))
        return js

    def me(self):
        r = requests.get(self.base_url + "/auth/me", headers=self.headers(), timeout=5)
        if r.status_code == 401:
            return None
        r.raise_for_status()
        return r.json().get("user")
