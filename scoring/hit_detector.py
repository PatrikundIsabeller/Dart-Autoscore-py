# hit_detector.py
from __future__ import annotations
import cv2 as cv, numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from board_model import BoardModel
from homography_utils import circle_world_to_poly, point_px_to_polar_mm, image_to_world

@dataclass
class Hit:
    camera_id: str
    tip_px: Tuple[float,float]
    tip_mm: Tuple[float,float]
    r_mm: float
    theta_deg: float
    ring: str
    sector: int
    score: int

class TipDetector:
    def __init__(self, board: BoardModel):
        self.board = board
        self.H_by_cam: Dict[str, np.ndarray] = {}
        self.bg_by_cam: Dict[str, np.ndarray] = {}
        self.last_hits_mm: Dict[str, List[np.ndarray]] = {}
        self.diff_thresh = 24
        self.min_area_px = 12
        self.max_area_px = 8000
        self.morph_open = 1
        self.morph_close = 3
        self.nms_mm = 8.0
        self.bg_update_alpha = 0.02
        self.calm_diff_sum_px = 1e6
    def set_homography(self, camera_id: str, H_world_to_image: np.ndarray) -> None:
        self.H_by_cam[camera_id] = np.asarray(H_world_to_image, dtype=np.float64)
    @staticmethod
    def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
        return cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    @staticmethod
    def _apply_morph(bin_img: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
        out = bin_img
        if open_k>0:
            k = cv.getStructuringElement(cv.MORPH_ELLIPSE,(open_k,open_k))
            out = cv.morphologyEx(out, cv.MORPH_OPEN, k)
        if close_k>0:
            k = cv.getStructuringElement(cv.MORPH_ELLIPSE,(close_k,close_k))
            out = cv.morphologyEx(out, cv.MORPH_CLOSE, k)
        return out
    @staticmethod
    def _weighted_centroid(patch: np.ndarray) -> Tuple[float,float]:
        ys,xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        w = patch.astype(np.float64) + 1.0
        cx = float((xs*w).sum()/w.sum()); cy = float((ys*w).sum()/w.sum())
        return cx, cy
    @staticmethod
    def _within_polygon_mask(shape_hw, poly: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape_hw, dtype=np.uint8); cv.fillPoly(mask, [poly], 255); return mask
    def _update_background(self, cam: str, gray: np.ndarray, diff_sum: float) -> None:
        if cam not in self.bg_by_cam:
            self.bg_by_cam[cam] = gray.astype(np.float32); return
        if diff_sum < self.calm_diff_sum_px:
            cv.accumulateWeighted(gray.astype(np.float32), self.bg_by_cam[cam], self.bg_update_alpha)
    def ingest_frame(self, camera_id: str, frame_bgr: np.ndarray) -> List[Hit]:
        assert camera_id in self.H_by_cam, "Homography not set for this camera"
        H = self.H_by_cam[camera_id]
        gray = self._to_gray(frame_bgr)
        poly = circle_world_to_poly(H, radius_mm=self.board.r_board_score_outer, n=180)
        mask_inside = self._within_polygon_mask(gray.shape, poly)
        if camera_id not in self.bg_by_cam: self.bg_by_cam[camera_id] = gray.astype(np.float32)
        bg8 = cv.convertScaleAbs(self.bg_by_cam[camera_id])
        diff = cv.absdiff(gray, bg8)
        diff_sum = float(diff.sum())
        self._update_background(camera_id, gray, diff_sum)
        _, bw = cv.threshold(diff, self.diff_thresh, 255, cv.THRESH_BINARY)
        bw &= mask_inside
        bw = self._apply_morph(bw, self.morph_open, self.morph_close)
        contours,_ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        hits: List[Hit] = []
        for c in contours:
            area = float(cv.contourArea(c))
            if area < self.min_area_px or area > self.max_area_px: continue
            comp_mask = np.zeros_like(bw); cv.drawContours(comp_mask,[c],-1,255,thickness=cv.FILLED)
            comp_inside = cv.bitwise_and(comp_mask, mask_inside)
            ys,xs = np.nonzero(comp_inside)
            if len(xs)==0: continue
            pts_px = np.stack([xs,ys], axis=1).astype(np.float64)
            pts_mm = image_to_world(H, pts_px)
            r_mm = np.hypot(pts_mm[:,0], pts_mm[:,1])
            i_min = int(np.argmin(r_mm))
            tip_px = (float(xs[i_min]), float(ys[i_min]))
            x0,y0 = int(round(tip_px[0])), int(round(tip_px[1]))
            x1,y1 = max(0,x0-1), max(0,y0-1)
            x2,y2 = min(diff.shape[1]-1,x0+1), min(diff.shape[0]-1,y0+1)
            patch = diff[y1:y2+1, x1:x2+1]
            if patch.size>=4:
                cx,cy = self._weighted_centroid(patch); tip_px = (x1+cx, y1+cy)
            r,theta = point_px_to_polar_mm(H, tip_px)
            score,ring,sector = self.board.score(r, theta)
            keep = True
            last_list = self.last_hits_mm.setdefault(camera_id, [])
            for prev in last_list:
                if np.hypot(prev[0]-pts_mm[i_min,0], prev[1]-pts_mm[i_min,1]) < self.nms_mm: keep=False; break
            if keep:
                last_list.append(pts_mm[i_min])
                hits.append(Hit(camera_id, tip_px, (float(pts_mm[i_min,0]), float(pts_mm[i_min,1])), float(r), float(theta), ring, sector, score))
        if len(self.last_hits_mm.get(camera_id, []))>16:
            self.last_hits_mm[camera_id] = self.last_hits_mm[camera_id][-16:]
        return hits
