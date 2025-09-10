# mock_agent.py – FastAPI-Mock für Controller/Calibration (eine Variante)
from __future__ import annotations

from fastapi import FastAPI, Request
import uvicorn, psutil, time, os, json
from pathlib import Path

APP_VERSION = "v0.1.0"
DET_VERSION = "v0.26.15"

# Persistenz: im Projektordner neben der .py
BASE_DIR = Path(__file__).resolve().parent
CALIB_PATH = BASE_DIR / "calibration.json"

# Default-Kalibrierung
CALIB = {
    "boardDiameterMm": 451.0,
    "rings": {
        "double_outer": 0.995, "double_inner": 0.940,
        "triple_outer": 0.620, "triple_inner": 0.540,
        "bull_outer": 0.060, "bull_inner": 0.015
    },
    "homographies": [],  # Liste von 3x3-Matrizen
    "intrinsics": []     # Liste von {k1,k2,enabled} pro Kamera
}

# Beim Start laden (falls vorhanden)
if CALIB_PATH.exists():
    try:
        CALIB.update(json.loads(CALIB_PATH.read_text(encoding="utf-8")))
        print(f"[mock_agent] Loaded calibration from {CALIB_PATH}")
    except Exception as e:
        print(f"[mock_agent] WARN cannot load {CALIB_PATH}: {e}")

app = FastAPI(title="TripleOne Mock Agent", version=APP_VERSION)
STATE = {"state": "THROW", "since": time.time()}


@app.get("/")
def root():
    return {"ok": True, "app": "TripleOne Mock Agent", "version": APP_VERSION}

@app.get("/health")
def health():
    p = psutil.Process()
    return {
        "ok": True,
        "appVersion": APP_VERSION,
        "detectorVersion": DET_VERSION,
        "uptime_s": int(time.time() - STATE["since"]),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "mem": {
            "rss": p.memory_info().rss,
            "vms": p.memory_info().vms,
        },
    }

# --- Calibration API ---------------------------------------------------------

@app.get("/calibration")
def get_calibration():
    return CALIB

@app.put("/calibration")
async def put_calibration(req: Request):
    body = await req.json()
    if not isinstance(body, dict):
        return {"ok": False, "error": "invalid json"}

    # Minimal-Validation + Merge
    if "boardDiameterMm" in body:
        CALIB["boardDiameterMm"] = float(body["boardDiameterMm"])
    if "rings" in body and isinstance(body["rings"], dict):
        CALIB["rings"] = body["rings"]
    if "homographies" in body and isinstance(body["homographies"], list):
        CALIB["homographies"] = body["homographies"]
    if "intrinsics" in body and isinstance(body["intrinsics"], list):
        CALIB["intrinsics"] = body["intrinsics"]

    CALIB_PATH.write_text(json.dumps(CALIB, indent=2), encoding="utf-8")
    return {"ok": True, "saved": str(CALIB_PATH)}

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4700, reload=False)
