# autodarts_geom.py
import numpy as np, cv2

R_BOARD_MM = 451.0 / 2.0
REL = {
    "bull_outer": 0.060,
    "bull_inner": 0.015,
    "triple_outer": 0.620,
    "triple_inner": 0.540,
    "double_inner": 0.940,
    "double_outer": 0.995,
}
R_DO_MM = REL["double_outer"] * R_BOARD_MM


def world_points_autodarts_mm() -> np.ndarray:
    # Reihenfolge: oben(20-1), rechts(6-10), unten(3-19), links(11-14)
    return np.array(
        [[0.0, -R_DO_MM], [R_DO_MM, 0.0], [0.0, R_DO_MM], [-R_DO_MM, 0.0]],
        dtype=np.float32,
    )


def compute_H_world2img(points_img_xy: np.ndarray) -> np.ndarray:
    assert points_img_xy.shape == (4, 2), "4 Bildpunkte erwartet"
    pts_world = world_points_autodarts_mm()
    H, _ = cv2.findHomography(pts_world, points_img_xy, method=0)
    if H is None:
        raise ValueError("Homography (Welt→Bild) fehlgeschlagen")
    return H


def M_tplPx_to_world_mm(
    R_tpl_px: float = 300.0, R_board_mm: float = R_BOARD_MM
) -> np.ndarray:
    s = R_board_mm / R_tpl_px  # mm/px (Template→mm)
    t = -s * R_tpl_px  # verschiebt Template-Mitte (300,300) nach (0,0)
    return np.array([[s, 0, t], [0, s, t], [0, 0, 1]], dtype=np.float64)


def templatePx_to_image_H(H_world2img: np.ndarray) -> np.ndarray:
    return H_world2img @ M_tplPx_to_world_mm()


def refine_points_on_double_outer(
    gray: np.ndarray, pts_img: np.ndarray, search_px: int = 6
) -> np.ndarray:
    c = np.mean(pts_img, axis=0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]
    out = []
    for x, y in pts_img:
        v = np.array([x, y]) - c
        n = np.linalg.norm(v) + 1e-6
        u = v / n
        best, best_s = (x, y), -1e9
        for d in range(-search_px, search_px + 1):
            xx = int(np.clip(x + u[0] * d, 1, w - 2))
            yy = int(np.clip(y + u[1] * d, 1, h - 2))
            s = gx[yy, xx] * u[0] + gy[yy, xx] * u[1]
            if s > best_s:
                best_s, best = float(s), (float(xx), float(yy))
        out.append(best)
    return np.array(out, dtype=np.float32)


def compute_homographies_from_points(gray: np.ndarray, pts_img_xy: np.ndarray):
    """Bequemer Einpunkt: liefert (H_world2img, H_tplPx2img, pts_refined)."""
    pts_ref = refine_points_on_double_outer(gray, pts_img_xy)
    H_world = compute_H_world2img(pts_ref)
    H_tpl = templatePx_to_image_H(H_world)
    return H_world, H_tpl, pts_ref
