# homography_utils.py
from __future__ import annotations
import numpy as np, cv2 as cv
from typing import Tuple
Mat3x3 = np.ndarray
def ensure_mat3x3(H: np.ndarray) -> Mat3x3:
    H = np.asarray(H, dtype=np.float64)
    assert H.shape == (3,3), "Homography must be 3x3"
    return H
def world_to_image(H: Mat3x3, pts_world_mm: np.ndarray) -> np.ndarray:
    H = ensure_mat3x3(H)
    pts = np.asarray(pts_world_mm, dtype=np.float64)
    P = np.hstack([pts, np.ones((pts.shape[0],1))])
    Q = (H @ P.T).T; Q /= Q[:,2:3]
    return Q[:, :2]
def image_to_world(H: Mat3x3, pts_image_px: np.ndarray) -> np.ndarray:
    H = ensure_mat3x3(H); Hinv = np.linalg.inv(H)
    pts = np.asarray(pts_image_px, dtype=np.float64)
    P = np.hstack([pts, np.ones((pts.shape[0],1))])
    Q = (Hinv @ P.T).T; Q /= Q[:,2:3]
    return Q[:, :2]
def circle_world_to_poly(H: Mat3x3, center_mm=(0.0,0.0), radius_mm=170.0, n=180) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_w = np.stack([center_mm[0] + radius_mm*np.cos(angles), center_mm[1] + radius_mm*np.sin(angles)], axis=1)
    pts_i = world_to_image(H, pts_w)
    return pts_i.reshape(-1,1,2).astype(np.int32)
def point_px_to_polar_mm(H: Mat3x3, pt_px: Tuple[float,float]) -> Tuple[float,float]:
    xy_mm = image_to_world(H, np.array([pt_px], dtype=np.float64))[0]
    x,y = float(xy_mm[0]), float(xy_mm[1])
    r = float(np.hypot(x,y)); theta = float(np.degrees(np.arctan2(y,x)) % 360.0)
    return r, theta
