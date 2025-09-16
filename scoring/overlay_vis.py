# overlay_vis.py
import cv2 as cv
from typing import List
from hit_detector import Hit
COLORS = {"inner_bull": (0,255,0), "outer_bull": (0,200,0), "single": (255,255,0), "double": (0,165,255), "treble": (255,0,0)}
def draw_hits(frame_bgr, hits: List[Hit]):
    for h in hits:
        c = COLORS.get(h.ring, (255,255,255))
        x, y = int(round(h.tip_px[0])), int(round(h.tip_px[1]))
        cv.circle(frame_bgr, (x,y), 4, c, thickness=-1)
        label = f"{h.score} ({h.ring} {h.sector if h.sector else ''})"
        cv.putText(frame_bgr, label, (x+6, y-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv.LINE_AA)
    return frame_bgr
