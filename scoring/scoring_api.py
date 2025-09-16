# scoring_api.py
from __future__ import annotations
import base64, time
from typing import List, Optional
from collections import deque
import cv2 as cv, numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from board_model import BoardModel
from hit_detector import TipDetector

app = FastAPI(title="TripleOne Scoring API", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
board = BoardModel(); engine = TipDetector(board)
_recent_hits = deque(maxlen=128)

class SetHomographyReq(BaseModel):
    camera_id: str; H: List[List[float]]
class DetectReq(BaseModel):
    camera_id: str; image_base64: str
class MapReq(BaseModel):
    camera_id: str; x: float; y: float

def _decode_b64_image(data: str) -> np.ndarray:
    if "," in data: data = data.split(",",1)[1]
    try: buf = base64.b64decode(data)
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
    arr = np.frombuffer(buf, dtype=np.uint8); img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Could not decode image")
    return img

@app.get("/health")
async def health(): return {"ok": True}

@app.post("/set_homography")
async def set_homography(req: SetHomographyReq):
    H = np.array(req.H, dtype=np.float64)
    if H.shape != (3,3): raise HTTPException(status_code=400, detail="H must be 3x3")
    engine.set_homography(req.camera_id, H); return {"ok": True}

@app.post("/detect")
async def detect(req: DetectReq):
    if req.camera_id not in engine.H_by_cam: raise HTTPException(status_code=400, detail="Set homography first")
    frame = _decode_b64_image(req.image_base64)
    hits = engine.ingest_frame(req.camera_id, frame)
    payload = []; now = time.time()
    for h in hits:
        item = {"ts": now, "camera_id": h.camera_id, "tip_px": [h.tip_px[0], h.tip_px[1]], "tip_mm": [h.tip_mm[0], h.tip_mm[1]],
                "r_mm": h.r_mm, "theta_deg": h.theta_deg, "ring": h.ring, "sector": h.sector, "score": h.score}
        payload.append(item); _recent_hits.append(item)
    return {"hits": payload}

@app.post("/map")
async def map_pixel(req: MapReq):
    from homography_utils import point_px_to_polar_mm
    if req.camera_id not in engine.H_by_cam: raise HTTPException(status_code=400, detail="Set homography first")
    H = engine.H_by_cam[req.camera_id]; r, theta = point_px_to_polar_mm(H, (req.x, req.y))
    score, ring, sector = board.score(r, theta)
    return {"r_mm": r, "theta_deg": theta, "ring": ring, "sector": sector, "score": score}

@app.get("/hits")
async def get_hits(camera_id: Optional[str]=Query(default=None), clear: bool=Query(default=True), max_items: int=Query(default=64, ge=1, le=128)):
    items = list(_recent_hits)
    if camera_id: items = [h for h in items if h["camera_id"] == camera_id]
    items = items[-max_items:]
    if clear:
        if camera_id:
            to_remove = [h for h in list(_recent_hits) if h["camera_id"] == camera_id]
            for h in to_remove:
                try: _recent_hits.remove(h)
                except ValueError: pass
        else: _recent_hits.clear()
    return {"hits": items, "count": len(items)}
