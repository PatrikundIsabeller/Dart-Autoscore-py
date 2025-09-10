# mock_agent.py  -- FastAPI-Mock für Controller-UI
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Request
import json
import uvicorn, psutil, time, random, os, json

app = FastAPI(title="TripleOne Mock Agent")
STATE = {"state": "THROW", "since": time.time()}
APP_VERSION = "v0.1.0"
DET_VERSION = "v0.26.15"
CAL_FILE = os.path.join(
    os.getenv("APPDATA", os.getcwd()), "TripleOne", "calibration.json"
)
os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)


class Calibration(BaseModel):
    boardDiameterMm: float = 451.0
    homographies: list[list[list[float]]] = []
    dist: list[float] | None = None
    rings: dict | None = None


@app.get("/status")
def get_status():
    # dummy werte
    fps = random.choice([29, 30, 31])
    cpu = psutil.cpu_percent(interval=0)
    mem = psutil.Process().memory_info().rss / 1_000_000
    return {
        "state": STATE["state"],
        "fps": fps,
        "cpu": cpu,
        "mem": round(mem, 2),
        "versions": {"app": APP_VERSION, "detector": DET_VERSION},
        "cams": [
            {"id": 0, "name": "CamL", "res": [1920, 1080], "fps": 30},
            {"id": 1, "name": "CamR", "res": [1920, 1080], "fps": 30},
            {"id": 2, "name": "CamB", "res": [1280, 720], "fps": 30},
        ],
    }


@app.post("/control/{action}")
def control(action: str):
    a = action.lower()
    if a == "restart":
        STATE["state"] = "ARMED"
    elif a == "stop":
        STATE["state"] = "IDLE"
    elif a == "reset":
        STATE["state"] = "ARMED"
    else:
        return {"ok": False, "msg": "unknown action"}
    STATE["since"] = time.time()
    return {"ok": True, "state": STATE["state"]}


@app.post("/calibrate/start")
def cal_start():
    STATE["state"] = "SETTLING"
    return {"ok": True}


@app.post("/calibrate/finish")
def cal_finish():
    STATE["state"] = "THROW"
    return {"ok": True}


@app.get("/calibration")
def cal_get():
    if os.path.exists(CAL_FILE):
        return json.load(open(CAL_FILE, "r", encoding="utf-8"))
    return {"boardDiameterMm": 451.0, "homographies": [], "rings": {}}


@app.put("/calibration")
async def put_calib(req: Request):
    payload = await req.json()
    print(">>> CALIB UPDATE\n", json.dumps(payload, indent=2))
    STATE["calibration"] = payload
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4700, reload=False)
