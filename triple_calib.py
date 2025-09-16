# triple_calib.py – 3-Kamera-Kalibrierung (PyQt6) – autodarts 4-Punkt, Welt→Bild Homographie
from __future__ import annotations
import os, json
from typing import List, Tuple, Optional

import cv2
import numpy as np
import requests
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QStandardPaths


# OpenCV Logging runterdrehen (sonst MSMF spammt die Konsole)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# =========================
# 1) Geometrie & Konstanten
# =========================

AGENT_URL = os.environ.get("TRIPLEONE_AGENT", "http://127.0.0.1:4700")

# autodarts-Relative Radien (unbedingt beibehalten)
REL = {
    "bull_outer": 0.060,
    "bull_inner": 0.015,
    "triple_outer": 0.620,
    "triple_inner": 0.540,
    "double_inner": 0.940,
    "double_outer": 0.995,
}
R_BOARD_MM = 451.0 / 2.0  # 225.5 mm
R_DO_MM = REL["double_outer"] * R_BOARD_MM

# Sektorreihenfolge im Uhrzeigersinn ab oben
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Wedge-Mitten-Offset (Segmentmitte statt Segmentgrenze)
WEDGE_SEGMENT_CENTER_OFFSET_DEG = 9.0


def world_points_autodarts_mm() -> np.ndarray:
    """
    4 Welt-Referenzpunkte (mm) auf dem Double-Outer – Kardinalachsen:
      1: oben (20-1)    2: rechts (6-10)
      3: unten (3-19)   4: links  (11-14)
    Reihenfolge: [oben, rechts, unten, links]
    """
    return np.array(
        [[0.0, -R_DO_MM], [R_DO_MM, 0.0], [0.0, R_DO_MM], [-R_DO_MM, 0.0]],
        dtype=np.float32,
    )


def refine_points_on_double_outer(
    gray: np.ndarray,
    pts_img: np.ndarray,
    search_px: int = 6,
    center: Optional[tuple[float, float]] = None,
    only_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Verfeinert die Punkte radial (vom festen 'center' aus) auf die Double-Outer-Kante.
    - center: fester Mittelpunkt (z.B. zum Drag-Beginn gemerkt). Fällt sonst auf Mittel der 4 Punkte zurück.
    - only_idx: None → alle; sonst nur dieser Index [0..3].
    Verschiebt NICHT ungefragt andere Punkte.
    """
    assert gray.ndim == 2 and pts_img.shape == (4, 2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]

    pts = pts_img.astype(np.float32).copy()
    c = (
        np.array(center, dtype=np.float32)
        if center is not None
        else np.mean(pts, axis=0)
    )
    idxs = range(4) if only_idx is None else [int(only_idx)]

    for i in idxs:
        x, y = map(float, pts[i])
        v = np.array([x, y], dtype=np.float32) - c
        n = float(np.linalg.norm(v) + 1e-6)
        u = v / n  # radialrichtung
        best, best_s = (x, y), -1e9
        for d in range(-search_px, search_px + 1):
            xx = int(np.clip(x + u[0] * d, 1, w - 2))
            yy = int(np.clip(y + u[1] * d, 1, h - 2))
            s = gx[yy, xx] * u[0] + gy[yy, xx] * u[1]
            if s > best_s:
                best_s, best = float(s), (float(xx), float(yy))
        pts[i] = best
    return pts


def compute_H_world2img(points_img_xy: np.ndarray) -> np.ndarray:
    """
    Welt→Bild Homographie aus den 4 Bildpunkten (oben/rechts/unten/links) auf dem Double-Outer.
    """
    assert points_img_xy.shape == (4, 2)
    pts_world = world_points_autodarts_mm()
    H, _ = cv2.findHomography(
        pts_world, points_img_xy, method=0
    )  # 4 exakt → kein RANSAC
    if H is None:
        raise ValueError("Homography (Welt→Bild) fehlgeschlagen")
    return H


def M_tplPx_to_world_mm(
    R_tpl_px: float = 300.0, R_board_mm: float = R_BOARD_MM
) -> np.ndarray:
    """
    Projektivmatrix: Template-Pixel (600×600, Mittelpunkt (300,300)) → Welt (mm, Mittelpunkt (0,0)).
    """
    s = R_board_mm / R_tpl_px  # mm/px (Template→mm)
    t = -s * R_tpl_px  # verschiebt (300,300) → (0,0)
    return np.array([[s, 0, t], [0, s, t], [0, 0, 1]], dtype=np.float64)


def templatePx_to_image_H(H_world2img: np.ndarray) -> np.ndarray:
    """
    Kombinierte Homographie: Template(600px) → Bild(px), via Welt(mm).
    """
    return H_world2img @ M_tplPx_to_world_mm()


def compute_homographies_from_points(gray: np.ndarray, pts_img_xy: np.ndarray):
    """
    Bequem-Funktion:
      - verfeinert die 4 Punkte auf der Double-Outer-Kante,
      - berechnet H_world2img und H_tplPx2img.
    Rückgabe: (H_world2img, H_tplPx2img, pts_refined)
    """
    pts_ref = refine_points_on_double_outer(gray, pts_img_xy.astype(np.float32))
    H_world = compute_H_world2img(pts_ref)
    H_tpl = templatePx_to_image_H(H_world)
    return H_world, H_tpl, pts_ref


# ===========================================
# 1b) warpPolar-basierter Rotations-Estimator
# ===========================================


def _wrap_deg(x: float) -> float:
    # -> [-180, 180)
    while x >= 180.0:
        x -= 360.0
    while x < -180.0:
        x += 360.0
    return x


def _wrap_mod_deg(x: float, mod: float = 18.0) -> float:
    # wrap in (-mod/2, +mod/2]
    y = (x + mod / 2.0) % mod - mod / 2.0
    return y if y != -mod / 2.0 else mod / 2.0


def estimate_rotation_mod18(
    gray: np.ndarray,
    center: tuple[float, float],
    r_equiv_px: float,
    rel_r0: float = REL["triple_inner"],
    rel_r1: float = REL["triple_outer"],
    ang_res: int = 1440,  # 0.25° Auflösung
    rad_res: int = 256,
) -> tuple[float, float]:
    """
    Liefert (rot_offset_deg_mod18, confidence).
    Idee:
      1) warpPolar um 'center'
      2) Summiere Kantenenergie im Triple-Band → Winkelprofil A(θ)
      3) FFT: 20. Harmonische → Phase → Segmentgrenzen-Phase
      4) Referenzgrenze oben (20/1) bei -81° (=-90°+9°) → delta = θ_bound - (-81°), modulo 18°
    """
    cx, cy = center
    maxR = max(32.0, float(r_equiv_px) * (REL["double_outer"] * 1.15))

    polar = cv2.warpPolar(
        gray,
        (ang_res, rad_res),
        (float(cx), float(cy)),
        maxR,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    r0_pix = float(r_equiv_px) * rel_r0
    r1_pix = float(r_equiv_px) * rel_r1
    i0 = int(np.clip(r0_pix / maxR * (rad_res - 1), 0, rad_res - 1))
    i1 = int(np.clip(r1_pix / maxR * (rad_res - 1), 0, rad_res - 1))
    if i1 <= i0:
        i1 = min(rad_res - 1, i0 + 1)

    band = polar[i0:i1, :]  # [rad, ang]
    if band.shape[0] >= 2:
        d = np.abs(np.diff(band.astype(np.float32), axis=0))
        A = d.sum(axis=0)
    else:
        A = band.astype(np.float32).sum(axis=0)

    A = cv2.GaussianBlur(A[None, :], (1, 31), 0).ravel()
    A = A - float(A.mean() if A.size else 0.0)

    N = int(A.size)
    if N < 64:
        return 0.0, 0.0
    F = np.fft.rfft(A)
    k = 20
    if k >= F.size:
        return 0.0, 0.0
    c20 = F[k]
    theta_bound_rad = -np.angle(c20) / 20.0
    theta_bound_deg = _wrap_deg(np.degrees(theta_bound_rad))

    ref_top_boundary_deg = -81.0
    delta = theta_bound_deg - ref_top_boundary_deg
    rot_mod18 = _wrap_mod_deg(delta, 18.0)

    mag = np.abs(F)
    upper = min(40, mag.size - 1)
    denom = mag[1:upper].sum() + 1e-6
    conf = float(np.abs(c20) / denom)
    conf = max(0.0, min(1.0, conf / 3.0))
    return float(rot_mod18), conf


# ===========================
# 2) HTTP I/O (Agent-Persistenz)
# ===========================


def fetch_calibration() -> dict:
    try:
        r = requests.get(AGENT_URL.rstrip("/") + "/calibration", timeout=1.5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def save_calibration(payload: dict) -> None:
    r = requests.put(AGENT_URL.rstrip("/") + "/calibration", json=payload, timeout=3.0)
    r.raise_for_status()


# ===========================
# 3) Overlay (Template 600px)
# ===========================

# ===========================
# 3) Overlay (Template 600px)
# ===========================


def make_grid600_image_colored(
    include_numbers: bool = True, fill: bool = False, as_rgba: bool = True
) -> np.ndarray:
    """
    600×600 Kalibrierungs-Overlay.
    - as_rgba=True: transparenter Hintergrund, Linien/Zahlen mit Alpha
    - fill=False: keine Farbflächen (nur Drahtgitter)
    """
    ch = 4 if as_rgba else 3
    img = np.zeros((600, 600, ch), dtype=np.uint8)
    cx = cy = 300
    R = 300.0

    # Farben (BGRA/BGR)
    def C(b, g, r, a=255):
        return (b, g, r, a) if ch == 4 else (b, g, r)

    col_line = C(255, 255, 255, 255)  # Weiß
    col_number = C(255, 255, 255, 255)
    col_shadow = C(0, 0, 0, 255)

    # Ringe (nur Linien)
    for rel, th in [
        (REL["double_outer"], 2),
        (REL["double_inner"], 2),
        (REL["triple_outer"], 2),
        (REL["triple_inner"], 2),
        (REL["bull_outer"], 2),
        (REL["bull_inner"], 2),
    ]:
        cv2.circle(img, (cx, cy), int(rel * R), col_line, th, cv2.LINE_AA)

    # 20 Sektorgrenzen
    r_bo = int(REL["bull_outer"] * R)
    r_do = int(REL["double_outer"] * R)
    for k in range(20):
        t = np.deg2rad(-90 + k * 18)
        x0, y0 = int(cx + r_bo * np.cos(t)), int(cy + r_bo * np.sin(t))
        x1, y1 = int(cx + r_do * np.cos(t)), int(cy + r_do * np.sin(t))
        cv2.line(img, (x0, y0), (x1, y1), col_line, 2, cv2.LINE_AA)

    # Zahlen in Segmentmitte (+9°)
    if include_numbers:
        r_txt = int(REL["double_outer"] * R * 1.06)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        for k, num in enumerate(SECTOR_ORDER):
            t = np.deg2rad(-90 + (k * 18 + 9))
            x = cx + r_txt * np.cos(t)
            y = cy + r_txt * np.sin(t)
            text = str(num)
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            org = (int(x - tw / 2), int(y + th / 2))
            # kleiner Shadow für Lesbarkeit
            cv2.putText(
                img,
                text,
                (org[0] + 1, org[1] + 1),
                font,
                scale,
                col_shadow,
                thick + 2,
                cv2.LINE_AA,
            )
            cv2.putText(img, text, org, font, scale, col_number, thick, cv2.LINE_AA)
    return img


# >>> beim Testen NUR Linien (keine blaue Füllung):
GRID_600_RGBA = make_grid600_image_colored(
    include_numbers=True, fill=False, as_rgba=True
)
GRID_600 = GRID_600_RGBA  # Alias, falls alter Name noch irgendwo auftaucht


# ==================
# 4) Camera Utilities
# ==================


def open_capture(cam_id: int, res=(1280, 720), fps: int = 30):
    """
    Öffnet eine Kamera robust:
      - bevorzugt DirectShow auf Windows (stabiler als MSMF bei vielen UVC-Cams)
      - setzt FOURCC=MJPG (häufig nötig für >=720p/30)
      - probiert Wunsch-Res/FPS, dann Fallbacks
      - führt Warmup-Reads durch; gibt nur ein cap zurück, das wirklich Frames liefert
    """
    # Backend-Reihenfolge: DSHOW -> MSMF -> ANY
    backends = []
    if os.name == "nt":
        backends = [
            getattr(cv2, "CAP_DSHOW", 700),
            getattr(cv2, "CAP_MSMF", 1400),
            getattr(cv2, "CAP_ANY", 0),
        ]
    else:
        # Linux/mac: ANY reicht meist; V4L2 ist default
        backends = [getattr(cv2, "CAP_ANY", 0)]

    # Fallback-Kombis: Wunsch, dann 1280x720@30, dann 640x480@30
    w_req, h_req = int(res[0]), int(res[1])
    combos = [(w_req, h_req, int(fps)), (1280, 720, 30), (640, 480, 30)]

    for be in backends:
        for w, h, f in combos:
            cap = None
            try:
                cap = cv2.VideoCapture(cam_id, be)
                if not cap or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue

                # Viele Cams brauchen MJPG, sonst kein Stream / falsche FPS
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

                # Erst geringe Default-Res, dann Ziel-Res (manche Treiber mögen die Sprünge nicht)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # kurz anwärmen
                ok = False
                for _ in range(6):
                    ok, _frm = cap.read()
                    if ok:
                        break
                if not ok:
                    cap.release()
                    continue

                # jetzt gewünschte Res/FPS setzen
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, f)

                # nochmal anwärmen und überprüfen
                ok = False
                for _ in range(10):
                    ok, _frm = cap.read()
                    if ok and _frm is not None and _frm.size > 0:
                        break
                if not ok:
                    cap.release()
                    continue

                # Alles gut – dieses cap benutzen
                return cap

            except Exception:
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                continue

    return None


def enumerate_cameras(max_idx: int = 10) -> list[int]:
    found = []
    for i in range(max_idx):
        cap = open_capture(i, (640, 480), 15)
        if cap:
            found.append(i)
            cap.release()
    return found


def cvimg_to_qpix(img: np.ndarray) -> QtGui.QPixmap:
    if img is None or img.size == 0:
        return QtGui.QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QPixmap.fromImage(
        QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
    )


def cvimg_to_qpix_any(img: np.ndarray) -> QtGui.QPixmap:
    if img is None or img.size == 0:
        return QtGui.QPixmap()
    h, w = img.shape[:2]
    if img.ndim == 2:  # Gray
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
    elif img.shape[2] == 3:  # BGR -> RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    elif img.shape[2] == 4:  # BGRA -> RGBA (Alpha erhalten!)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888)
    else:
        return QtGui.QPixmap()
    return QtGui.QPixmap.fromImage(qimg)


def undistort_simple(frame: np.ndarray, k1: float, k2: float):
    """Grobe Radial-Entzerrung (k1,k2). Reicht für Webcams meist aus."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    fx = fy = float(max(w, h))
    cx, cy = w * 0.5, h * 0.5
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    return cv2.undistort(frame, K, D, None, newK)


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    if lo == hi:
        lo = max(0, lo - 10)
        hi = min(255, hi + 10)
    return lo, hi


def _prep_gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    return g


def detect_board_circle(gray: np.ndarray) -> Optional[tuple[float, float, float]]:
    """
    Liefert (cx, cy, r) wenn möglich. 3 Stufen: Hough ALT → Hough klassisch → Kontur-Fallback.
    """
    H, W = gray.shape[:2]
    r_min = int(0.28 * min(H, W))
    r_max = int(0.60 * min(H, W))
    # --- Pass 1: HOUGH_GRADIENT_ALT (OpenCV >=4.5), sehr robust ---
    try:
        circles = cv2.HoughCircles(
            gray,
            getattr(cv2, "HOUGH_GRADIENT_ALT", cv2.HOUGH_GRADIENT),
            dp=1.3,
            minDist=min(H, W) / 3,
            param1=120,
            param2=0.8,  # ALT: param2 ist Score [0..1]
            minRadius=r_min,
            maxRadius=r_max,
        )
        if circles is not None and len(circles[0]) > 0:
            c = circles[0][0]
            return float(c[0]), float(c[1]), float(c[2])
    except Exception:
        pass

    # --- Pass 2: klassisch ---
    lo, hi = _auto_canny(gray)
    edges = cv2.Canny(gray, lo, hi)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(H, W) / 3,
        param1=120,
        param2=30,
        minRadius=r_min,
        maxRadius=r_max,
    )
    if circles is not None and len(circles[0]) > 0:
        c = max(circles[0], key=lambda v: v[2])
        return float(c[0]), float(c[1]), float(c[2])

    # --- Pass 3: Kontur-Fallback ---
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.medianBlur(th, 5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in cnts:
        if len(c) < 20:
            continue
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri <= 1e-3:
            continue
        roundness = 4 * np.pi * area / (peri * peri)  # 1 = Kreis
        if area < 0.05 * (H * W):  # zu klein
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        # Score: groß & rund & im erlaubten Radiusfenster
        if r_min <= r <= r_max:
            score = area * (0.5 + 0.5 * roundness)
            if score > best_score:
                best_score, best = (float(cx), float(cy), float(r)), score
    return best


# =======================
# 5) DragBoard (Overlay UI)
# =======================


class DragBoard(QtWidgets.QLabel):
    """
    Liveframe + warped Grid + 4 draggable Punkte (oben,rechts,unten,links) auf Double-Outer.
    Wedge (20er-Sektor) wird im 600x600-Template erzeugt und via Homographie gemappt.
    """

    pointsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setStyleSheet(
            "QLabel{border:1px solid #404040;border-radius:8px;background:#111;color:#fff;}"
        )
        self.setMouseTracking(True)  # Move-Events auch ohne gedrückte Taste
        self._ref_center: Optional[tuple[float, float]] = (
            None  # fixer Mittelpunkt fürs Refinement
        )

        # ---- State ----
        self.frame: Optional[np.ndarray] = None
        # Startpunkte: grob um das Bild platziert (oben, rechts, unten, links)
        self.points = np.array(
            [[300, 120], [540, 300], [300, 480], [60, 300]], dtype=np.float32
        )
        self._drag_idx = -1
        self._active_idx = 0

        # Marker-Style
        self.pt_radius = 8  # Radius der Markerpunkte (px, im Widget)
        self.pt_thickness = 2  # Linienstärke
        self.cross_len = 7  # Länge der Crosshair-Linien (px)
        self.color_point = QtGui.QColor("#22c55e")  # grün
        self.color_active = QtGui.QColor("#f59e0b")  # amber für aktiven Punkt
        self.color_label = QtGui.QColor("#FF8800")  # Text Orange
        # --- Magnifier (Lupe) ---
        self._mag_active = False
        self._mag_last_img = (0.0, 0.0)  # (x_img, y_img)
        self._mag_size = 160  # Durchmesser der Lupe (px, on-screen)
        self._mag_zoom = 3.0  # Vergrößerungsfaktor
        self._mag_crosshair = True

        # Snapshot & Dirty-Flag
        self._snapshot_gray: Optional[np.ndarray] = None
        self._h_dirty = True  # True → H neu berechnen

        # Homographien
        self.H_world2img: Optional[np.ndarray] = None  # für Speichern/Agent
        self.H_tplPx2img: Optional[np.ndarray] = None  # fürs Overlay

        # Overlay Darstellung
        self.alpha = 0.65

        # Wedge im Template
        self._wedge_tmpl: Optional[np.ndarray] = None  # (N,1,2) float32
        self._wedge_angle_deg: float = 90.0  # 20 oben

    # ----- Wedge (Template) -----
    def set_wedge_template(
        self,
        angle_up_deg: float = 90.0,
        spread_deg: float = 18.0,
        r_in: float = REL["bull_outer"],
        r_out: float = REL["double_outer"],
    ) -> None:
        self._wedge_angle_deg = angle_up_deg
        cx = cy = 300.0
        R = 300.0
        a0 = np.deg2rad(angle_up_deg - spread_deg / 2.0)
        a1 = np.deg2rad(angle_up_deg + spread_deg / 2.0)
        outer = [
            (cx + R * r_out * np.cos(t), cy - R * r_out * np.sin(t))
            for t in np.linspace(a0, a1, 24)
        ]
        inner = [
            (cx + R * r_in * np.cos(t), cy - R * r_in * np.sin(t))
            for t in np.linspace(a1, a0, 24)
        ]
        self._wedge_tmpl = np.array(outer + inner, dtype=np.float32).reshape(-1, 1, 2)
        self.update()

    # ----- API -----
    def set_alpha(self, a: float):
        self.alpha = max(0.0, min(1.0, a))
        self.update()

    def set_frame(self, frame: Optional[np.ndarray]):
        self.frame = frame
        self.update()

    def set_points(self, pts: np.ndarray):
        if pts is not None and pts.shape == (4, 2):
            self.points = pts.astype(np.float32)
            self._h_dirty = True
            self.pointsChanged.emit()
            self.update()

    def set_snapshot_from_current_frame(self):
        """Nimmt ein Graustufen-Snapshot aus self.frame (falls vorhanden)."""
        if self.frame is not None and self.frame.size:
            self._snapshot_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def recompute_homography(
        self, refine: bool = False, only_idx: Optional[int] = None
    ):
        """
        Rechnet H auf Basis der aktuellen Punkte.
        - refine=False: keine Punktverschiebung (nur H neu)
        - refine=True: verfeinert nur 'only_idx' oder alle (falls None) – mit festem _ref_center
        """
        if self.points.shape != (4, 2):
            return
        gray = self._snapshot_gray
        if gray is None and self.frame is not None and self.frame.size:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if gray is None:
            return

        try:
            if refine:
                center = (
                    self._ref_center
                    if self._ref_center is not None
                    else tuple(np.mean(self.points, axis=0))
                )
                self.points = refine_points_on_double_outer(
                    gray, self.points, center=center, only_idx=only_idx
                )

            H_world = compute_H_world2img(self.points.astype(np.float32))
            H_tpl = templatePx_to_image_H(H_world)

            self.H_world2img = H_world
            self.H_tplPx2img = H_tpl
            self._h_dirty = False
        except Exception:
            self.H_world2img = None
            self.H_tplPx2img = None
            self._h_dirty = True

    def homography(self) -> Optional[np.ndarray]:
        """
        Kompatibilitäts-API: stellt sicher, dass H gültig ist, und gibt H_tplPx2img zurück.
        (Wird z.B. in CamPanel.current_overlay_h() / current_world_h() aufgerufen.)
        """
        if self.H_tplPx2img is None or self._h_dirty:
            self.set_snapshot_from_current_frame()
            self.recompute_homography(refine=True)
        return self.H_tplPx2img

    # ----- Painting -----
    def _paint_pixmap(self, p: QtGui.QPainter, pm: QtGui.QPixmap):
        rect = self.rect()
        ps = pm.size()
        scale = min(rect.width() / ps.width(), rect.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (rect.width() - dw) / 2.0, (rect.height() - dh) / 2.0
        p.drawPixmap(
            QtCore.QRectF(ox, oy, dw, dh),
            pm,
            QtCore.QRectF(0, 0, ps.width(), ps.height()),
        )
        return scale, ox, oy, ps

    def _img_to_widget(
        self,
        x_img: float,
        y_img: float,
        ox: float,
        oy: float,
        scale: float,
        ps: QtCore.QSize,
    ) -> tuple[float, float]:
        """Mapt Bildkoordinate (Pixel im Kameraframe) in Widgetkoordinate."""
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        x = ox + (x_img / fw) * (ps.width() * scale)
        y = oy + (y_img / fh) * (ps.height() * scale)
        return float(x), float(y)

    def _draw_magnifier(
        self, p: QtGui.QPainter, ox: float, oy: float, scale: float, ps: QtCore.QSize
    ):
        if not self._mag_active or self.frame is None or not self.frame.size:
            return

        xi, yi = self._mag_last_img
        dst_d = int(self._mag_size)  # Durchmesser on-screen
        zoom = float(self._mag_zoom)
        src_d = max(12, int(round(dst_d / zoom)))  # Quell-Kantenlänge (quadratisch)

        # --- ROI robust zentriert um (xi, yi) holen ---
        # getRectSubPix behandelt Out-of-bounds sauber; kein manuelles Clamp nötig
        try:
            patch = cv2.getRectSubPix(
                self.frame, (src_d, src_d), (float(xi), float(yi))
            )
        except cv2.error:
            # Fallback: falls Kamera wechselt etc.
            h, w = self.frame.shape[:2]
            x0 = int(round(max(0, min(w - src_d, xi - src_d * 0.5))))
            y0 = int(round(max(0, min(h - src_d, yi - src_d * 0.5))))
            patch = self.frame[y0 : y0 + src_d, x0 : x0 + src_d]

        # Hochskalieren (nearest = knackig)
        patch = cv2.resize(patch, (dst_d, dst_d), interpolation=cv2.INTER_NEAREST)
        pm = cvimg_to_qpix(patch)

        # Widget-Koordinate des Zielpunkts
        cx, cy = self._img_to_widget(xi, yi, ox, oy, scale, ps)

        # *** exakt zentriert unter dem Kreuz (kein Versatz) ***
        x_draw = cx - dst_d * 0.5
        y_draw = cy - dst_d * 0.5
        # kein Clamping mehr, damit die Mitte exakt übereinander liegt

        # runde Maske + leichte Umrandung
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QRectF(x_draw, y_draw, dst_d, dst_d))

        p.save()
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

            # Drop-Shadow
            shadow = QtGui.QColor(0, 0, 0, 120)
            p.fillPath(path.translated(2, 2), shadow)

            # Clipping und Patch malen
            p.setClipPath(path)
            p.drawPixmap(
                QtCore.QRectF(x_draw, y_draw, dst_d, dst_d),
                pm,
                QtCore.QRectF(0, 0, pm.width(), pm.height()),
            )
            p.setClipping(False)

            # Rand
            p.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 2))
            p.drawPath(path)

            # Crosshair in der Lupe
            if self._mag_crosshair:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1))
                cx2 = x_draw + dst_d * 0.5
                cy2 = y_draw + dst_d * 0.5
                L = dst_d * 0.45
                p.drawLine(QtCore.QPointF(cx2 - L, cy2), QtCore.QPointF(cx2 + L, cy2))
                p.drawLine(QtCore.QPointF(cx2, cy2 - L), QtCore.QPointF(cx2, cy2 + L))
                p.setBrush(QtGui.QColor(255, 255, 255, 220))
                p.drawEllipse(QtCore.QPointF(cx2, cy2), 2, 2)
        finally:
            p.restore()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        if self.frame is None or not self.frame.size:
            p.setPen(QtGui.QPen(QtGui.QColor("#aaa")))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No signal")
            return

        pm = cvimg_to_qpix(self.frame)
        scale, ox, oy, ps = self._paint_pixmap(p, pm)

        # --- 1) MARKER zuerst (hohler Kreis + Crosshair) ---
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        labels = ["20-1", "6-10", "3-19", "11-14"]
        for idx, (x_img, y_img) in enumerate(self.points):
            x = ox + (x_img / fw) * (ps.width() * scale)
            y = oy + (y_img / fh) * (ps.height() * scale)

            pen_color = (
                self.color_active if idx == self._active_idx else self.color_point
            )
            pen = QtGui.QPen(pen_color, self.pt_thickness)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

            # Hohler Kreis
            p.drawEllipse(
                QtCore.QPointF(float(x), float(y)), self.pt_radius, self.pt_radius
            )
            # Crosshair
            L = self.cross_len
            p.drawLine(
                QtCore.QPointF(float(x - L), float(y)),
                QtCore.QPointF(float(x + L), float(y)),
            )
            p.drawLine(
                QtCore.QPointF(float(x), float(y - L)),
                QtCore.QPointF(float(x), float(y + L)),
            )

            # Label
            p.setPen(QtGui.QPen(self.color_label))
            p.drawText(QtCore.QPointF(float(x) + 10.0, float(y) - 10.0), labels[idx])

        # --- 2) OVERLAY danach (liegt über Markern; Linien wirken nicht eingedellt) ---
        H_tpl = self.H_tplPx2img
        if H_tpl is not None:
            p.save()
            try:
                h, w = self.frame.shape[:2]
                warped = cv2.warpPerspective(
                    GRID_600_RGBA,
                    H_tpl,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),  # komplett transparent
                )
                wpm = cvimg_to_qpix_any(warped)

                # Overlay bringt seine eigene Alpha mit → keine globale Opacity nötig
                # (wenn du global dämpfen willst, setz z.B. p.setOpacity(0.9) davor)
                self._paint_pixmap(p, wpm)

                # Wedge
                if self._wedge_tmpl is not None:
                    try:
                        pts_img = cv2.perspectiveTransform(
                            self._wedge_tmpl, H_tpl
                        ).reshape(-1, 2)
                        fw, fh = self.frame.shape[1], self.frame.shape[0]
                        path = QtGui.QPainterPath()
                        first = True
                        for x_img, y_img in pts_img:
                            x = ox + (x_img / fw) * (ps.width() * scale)
                            y = oy + (y_img / fh) * (ps.height() * scale)
                            if first:
                                path.moveTo(float(x), float(y))
                                first = False
                            else:
                                path.lineTo(float(x), float(y))
                        path.closeSubpath()
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 64, 64, 220), 1))
                        p.setBrush(QtGui.QColor(255, 64, 64, 90))
                        p.drawPath(path)
                    except Exception:
                        pass
            finally:
                p.restore()

        # --- 3) Lupe IMMER zuletzt (on top) ---
        self._draw_magnifier(p, ox, oy, scale, ps)

    # ----- Maus/Tastatur -----
    def _img_coords_from_mouse(self, e: QtGui.QMouseEvent):
        pm = cvimg_to_qpix(self.frame)
        rect = self.rect()
        ps = pm.size()
        scale = min(rect.width() / ps.width(), rect.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (rect.width() - dw) / 2.0, (rect.height() - dh) / 2.0
        pos = e.position()
        x = (pos.x() - ox) / scale
        y = (pos.y() - oy) / scale
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        x_img = max(0, min(fw - 1, x * (fw / ps.width())))
        y_img = max(0, min(fh - 1, y * (fh / ps.height())))
        return x_img, y_img

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self.frame is None or not self.frame.size:
            return
        xi, yi = self._img_coords_from_mouse(e)
        d = np.linalg.norm(self.points - np.array([xi, yi]), axis=1)
        self._drag_idx = int(np.argmin(d)) if d.size else -1
        if self._drag_idx >= 0:
            self._active_idx = self._drag_idx
            # Mittelpunkt für den gesamten Drag einfrieren:
            self._ref_center = tuple(np.mean(self.points, axis=0))
        self._mag_active = self._drag_idx >= 0
        self._mag_last_img = (xi, yi)
        self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.frame is None or not self.frame.size:
            return
        if e.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._mag_active = True
        xi, yi = self._img_coords_from_mouse(e)
        self._mag_last_img = (xi, yi)
        if self._drag_idx < 0:
            self.update()
            return
        self.points[self._drag_idx] = [xi, yi]
        self.pointsChanged.emit()
        # H live neu (ohne Refinement → andere Punkte bleiben exakt)
        self.recompute_homography(refine=False)
        self.update()

    def mouseReleaseEvent(self, _e: QtGui.QMouseEvent):
        idx = self._drag_idx
        self._drag_idx = -1
        self._mag_active = False
        self.set_snapshot_from_current_frame()
        # Nur den eben gesetzten Punkt fein­snappen (optional; kannst du auch weglassen)
        self.recompute_homography(
            refine=True, only_idx=(idx if idx is not None and idx >= 0 else None)
        )
        self.update()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        k = e.key()
        step = 1.0
        if e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            step = 5.0
        if e.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            step = 20.0

        if k == QtCore.Qt.Key.Key_Left:
            self.points[self._active_idx, 0] -= step
        elif k == QtCore.Qt.Key.Key_Right:
            self.points[self._active_idx, 0] += step
        elif k == QtCore.Qt.Key.Key_Up:
            self.points[self._active_idx, 1] -= step
        elif k == QtCore.Qt.Key.Key_Down:
            self.points[self._active_idx, 1] += step
        elif k in (
            QtCore.Qt.Key.Key_1,
            QtCore.Qt.Key.Key_2,
            QtCore.Qt.Key.Key_3,
            QtCore.Qt.Key.Key_4,
        ):
            self._active_idx = int(k - QtCore.Qt.Key.Key_1)
            self.update()
            return
        elif k == QtCore.Qt.Key.Key_R:
            # 'R' = refine alle; Shift+R = nur aktiver
            self.set_snapshot_from_current_frame()
            only = (
                self._active_idx
                if (e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
                else None
            )
            self.recompute_homography(refine=True, only_idx=only)
            self.update()
            return
        else:
            return

        self.pointsChanged.emit()
        # Nach Pfeiltasten H neu ohne Snap
        self.recompute_homography(refine=False)
        self.update()


# ===================
# 6) CamPanel (1 Cam)
# ===================


class CamPanel(QtWidgets.QFrame):
    """
    Eine Kamera-Kachel:
      - Kamera wählen
      - Undistort (k1/k2)
      - Auto (grober Seed via Ellipse + warpPolar-Offset)
      - 4-Punkt-Klick (Double-Outer) → Welt→Bild Homographie
    """

    def __init__(self, cam_id_guess: int):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # Ansicht
        self.view = DragBoard()
        self.view.set_wedge_template(
            90.0 + WEDGE_SEGMENT_CENTER_OFFSET_DEG
        )  # 20-Mitte nach oben

        # self.view.set_wedge_template(90.0)  # 20 nach oben im Template
        # self.view.set_wedge_template(99.0)  # 20-MITTE nach oben (90° + 9°)

        # Kameraauswahl (Items füllt TripleCalibration)
        self.cmb = QtWidgets.QComboBox()
        self.cmb.addItem("Select camera", None)
        self.cmb.currentIndexChanged.connect(self._on_select)

        # Entzerrung (grob)
        self.cb_undist = QtWidgets.QCheckBox("Undistort")
        self.k1 = QtWidgets.QDoubleSpinBox()
        self.k1.setRange(-0.50, 0.50)
        self.k1.setDecimals(3)
        self.k1.setSingleStep(0.005)
        self.k1.setValue(0.000)
        self.k1.setPrefix("k1 ")
        self.k2 = QtWidgets.QDoubleSpinBox()
        self.k2.setRange(-0.50, 0.50)
        self.k2.setDecimals(3)
        self.k2.setSingleStep(0.005)
        self.k2.setValue(0.000)
        self.k2.setPrefix("k2 ")

        # Nudge-Buttons ±18° fürs Wedge (visuell)
        self.btn_left = QtWidgets.QToolButton()
        self.btn_left.setText("⟲")
        self.btn_right = QtWidgets.QToolButton()
        self.btn_right.setText("⟳")

        # Auto-Knopf
        self.btn_auto = QtWidgets.QPushButton("Auto")

        # UI-Leiste unten
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.cmb, 1)
        bar.addWidget(self.cb_undist, 0)
        bar.addWidget(self.k1, 0)
        bar.addWidget(self.k2, 0)
        bar.addWidget(self.btn_left, 0)
        bar.addWidget(self.btn_right, 0)
        bar.addWidget(self.btn_auto, 0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)

        # Signale
        def _on_undist_changed():
            # Nach Entzerrungswechsel H stabil neu berechnen
            self.view.set_snapshot_from_current_frame()
            if self.view.points.shape == (4, 2):
                self.view.recompute_homography(refine=True)
            self.view.update()

        self.cb_undist.stateChanged.connect(lambda *_: _on_undist_changed())
        self.k1.valueChanged.connect(lambda *_: _on_undist_changed())
        self.k2.valueChanged.connect(lambda *_: _on_undist_changed())

        self.btn_auto.clicked.connect(self.auto_calibrate)
        self.btn_left.clicked.connect(lambda: self._nudge_wedge(-18))
        self.btn_right.clicked.connect(lambda: self._nudge_wedge(+18))

        # Laufzeitstate
        self.cap: Optional[cv2.VideoCapture] = None
        self.res = (1280, 720)
        self.fps = 30
        self._last_frame: Optional[np.ndarray] = None

    # ---------- Kamera-Lifecycle ----------
    def _on_select(self, _):
        cam = self.cmb.currentData()
        if cam is None:
            self.stop()
        else:
            self.start(int(cam))

    def start(self, cam_id: int):
        self.stop()
        cap = open_capture(cam_id, self.res, self.fps)
        if cap is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Camera",
                f"Cannot open camera {cam_id} with any backend.\n"
                f"• Prüfe Windows > Datenschutz > Kamera (Desktop-Apps zulassen)\n"
                f"• Schließe andere Programme (OBS/Teams/Browser), die die Cam blockieren",
            )
            return
        # optional: eine Probe lesen
        ok, frm = cap.read()
        if not ok or frm is None or frm.size == 0:
            cap.release()
            QtWidgets.QMessageBox.warning(
                self, "Camera", f"Camera {cam_id} opened, but no frames."
            )
            return
        self.cap = cap

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.view.set_frame(None)
        self._last_frame = None

    def tick(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.view.set_frame(None)
            return
        if self.cb_undist.isChecked():
            frame = undistort_simple(
                frame, float(self.k1.value()), float(self.k2.value())
            )
        self._last_frame = frame.copy()
        self.view.set_frame(frame.copy())

    def set_globals(self, res: tuple[int, int], fps: int):
        self.res = res
        self.fps = fps
        if self.cap:
            idx = self.cmb.currentData()
            if idx is not None:
                self.start(int(idx))

    def current_overlay_h(self) -> Optional[np.ndarray]:
        return self.view.H_tplPx2img

    def current_world_h(self) -> Optional[np.ndarray]:
        return self.view.H_world2img

    def current_points(self) -> list[list[float]]:
        return self.view.points.tolist()

    # ---------- Auto (Ellipse + warpPolar) ----------
    def auto_calibrate(self):
        """
        Robust:
        - optional undistort
        - Board-Kreis finden (3-Stufen-Fallback)
        - warpPolar/FFT → Rotationsoffset (mod 18°)
        - 4 Punkte auf Double-Outer setzen (Kreisannahme)
        - einmaliges Refinement aller 4 (fixierter Center) → H berechnen
        """
        # Kamera sicherstellen
        if not self.cap:
            cam = self.cmb.currentData()
            if cam is None:
                QtWidgets.QMessageBox.information(
                    self, "Auto", "Bitte zuerst eine Kamera wählen."
                )
                return
            self.start(int(cam))
            if not self.cap:
                return

        ok, frame = self.cap.read()
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Auto", "Kein Kamerabild verfügbar.")
            return

        # optional undistort, damit die Kreissuche einfacher wird
        if self.cb_undist.isChecked():
            frame = undistort_simple(
                frame, float(self.k1.value()), float(self.k2.value())
            )

        gray = _prep_gray(frame)

        found = detect_board_circle(gray)
        if not found:
            QtWidgets.QMessageBox.warning(
                self,
                "Auto",
                "Board-Kreis nicht gefunden.\nLicht & Kontrast prüfen, Kamera näher/zentrischer ausrichten.",
            )
            return

        cx, cy, r = found

        # Rotationsoffset (mod 18°) via warpPolar im Triple-Band
        try:
            rot_mod18, conf = estimate_rotation_mod18(gray, (cx, cy), r)
        except Exception:
            rot_mod18, conf = 0.0, 0.0

        # Basis-Grenzwinkel (20/1 oben = -81°, dann +90° Schritte)
        base = np.array([-81.0, 9.0, 99.0, -171.0], dtype=np.float32) + float(rot_mod18)

        # Punkte auf Double-Outer (Kreisannahme); Refinement zieht sie später exakt auf die Kante
        r_do_px = float(r) * REL["double_outer"]
        pts = []
        for deg in base:
            t = np.deg2rad(deg)
            x = cx + r_do_px * np.cos(t)
            y = cy + r_do_px * np.sin(t)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.float32)

        # In die View eintragen
        self.view.set_points(pts)
        self.view.set_frame(
            frame.copy()
        )  # sichert, dass Snapshot auf gleichem Bild basiert
        self.view.set_snapshot_from_current_frame()
        # fixen Mittelpunkt fürs Refinement setzen (verhindert „Wandern“)
        self.view._ref_center = (cx, cy)
        # einmalig alle 4 fein einschnappen & Homographien berechnen
        self.view.recompute_homography(refine=True, only_idx=None)
        self.view.update()

        if conf < 0.35:
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                f"Auto-Rotation unsicher (conf={conf:.2f}). Punkte kurz prüfen/feintunen.",
                self,
            )

    def _nudge_wedge(self, delta_deg: float):
        """Dreht *nur* das Wedge-Template (±18° / Feinnudge)."""
        if hasattr(self.view, "_wedge_angle_deg"):
            self.view.set_wedge_template(self.view._wedge_angle_deg + delta_deg)

    # def _appdata_calib_path(filename: str = "calibration_last.json") -> str:
    #   base = QStandardPaths.writableLocation(
    #      QStandardPaths.StandardLocation.AppDataLocation
    # )
    # if not base:
    #   base = os.path.join(os.path.expanduser("~"), ".tripleone")
    # os.makedirs(base, exist_ok=True)
    # return os.path.join(base, filename)


# ===========================
# 7) Hauptdialog / Save/Load
# ===========================


class TripleCalibration(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure cameras")
        self.resize(1280, 760)

        title = QtWidgets.QLabel("Configure cameras")
        title.setStyleSheet("font-size:24px;font-weight:700;margin:8px 0 16px 0;")

        # --- Panels (3 Cams) ---
        self.panels = [CamPanel(0), CamPanel(1), CamPanel(2)]

        # nur vorhandene Kameras ins Dropdown
        avail = enumerate_cameras(10)
        for p in self.panels:
            p.cmb.blockSignals(True)
            p.cmb.clear()
            p.cmb.addItem("Select camera", None)
            for idx in avail:
                p.cmb.addItem(f"Cam {idx}", idx)
            p.cmb.blockSignals(False)

        row = QtWidgets.QHBoxLayout()
        for p in self.panels:
            row.addWidget(p, 1)

        # --- Globals (Res/FPS) + Actions ---
        self.cmb_res = QtWidgets.QComboBox()
        for w, h in [(1920, 1080), (1600, 900), (1280, 720), (640, 480)]:
            self.cmb_res.addItem(f"{w}x{h}", (w, h))
        self.cmb_res.setCurrentIndex(2)

        self.cmb_fps = QtWidgets.QComboBox()
        for f in [30, 25, 20, 15]:
            self.cmb_fps.addItem(str(f), f)

        self.btn_save_agent = QtWidgets.QPushButton("Save to Agent")
        self.btn_save_file = QtWidgets.QPushButton("Save to File")
        self.btn_load_file = QtWidgets.QPushButton("Load from File")
        self.btn_save_default = QtWidgets.QPushButton("Save as Default")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Camera resolution"), 0, 0)
        grid.addWidget(self.cmb_res, 1, 0)
        grid.addWidget(QtWidgets.QLabel("Frames per second"), 0, 1)
        grid.addWidget(self.cmb_fps, 1, 1)
        grid.addItem(
            QtWidgets.QSpacerItem(
                40,
                20,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            1,
            2,
        )
        grid.addWidget(self.btn_load_file, 1, 3)
        grid.addWidget(self.btn_save_file, 1, 4)
        grid.addWidget(self.btn_save_agent, 1, 5)
        grid.addWidget(self.btn_save_default, 1, 6)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(title)
        root.addLayout(row)
        root.addSpacing(8)
        root.addLayout(grid)

        # --- wiring ---
        self.cmb_res.currentIndexChanged.connect(self._apply_globals)
        self.cmb_fps.currentIndexChanged.connect(self._apply_globals)
        self.btn_save_agent.clicked.connect(self._save_agent)
        self.btn_save_file.clicked.connect(self._save_file)
        self.btn_load_file.clicked.connect(self._load_file)
        self.btn_save_default.clicked.connect(self._save_last_session)

        # --- timer (UI tick) ---
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

        # initiale Capture-Settings auf Panels pushen
        self._apply_globals()

        # Auto-Load der letzten Session (falls vorhanden)
        self._load_last_session()

    # -------- helpers --------
    @staticmethod
    def _appdata_calib_path(filename: str = "calibration_last.json") -> str:
        # nutzt Qt Standardpfad; fällt auf ~/.tripleone zurück
        base = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppDataLocation
        )
        if not base:
            base = os.path.join(os.path.expanduser("~"), ".tripleone")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, filename)

    def _apply_globals(self):
        res = self.cmb_res.currentData()
        fps = int(self.cmb_fps.currentData())
        for p in self.panels:
            p.set_globals(res, fps)

    def _tick(self):
        for p in self.panels:
            p.tick()

    # -------- Save/Load (Datei) --------
    def _save_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration", "calibration.json", "JSON (*.json)"
        )
        if not path:
            return

        data = {}
        for i, p in enumerate(self.panels, start=1):
            # Sicherstellen, dass H befüllt ist (ohne Refinement)
            p.view.set_snapshot_from_current_frame()
            p.view.recompute_homography(refine=False)

            H_tpl = p.current_overlay_h()
            H_w = p.current_world_h()
            if H_tpl is None or H_w is None:
                continue
            data[f"camera_{i}"] = {
                "points_img": p.current_points(),
                "H_overlay_tplPx2img": H_tpl.tolist(),
                "H_world2img": H_w.tolist(),
                "intrinsics": {
                    "k1": float(p.k1.value()),
                    "k2": float(p.k2.value()),
                    "enabled": bool(p.cb_undist.isChecked()),
                },
                "board_diameter_mm": 451.0,
                "rel_radii": REL,
                "overlay": {
                    "alpha": float(p.view.alpha),
                    "wedge_angle_deg": float(getattr(p.view, "_wedge_angle_deg", 90.0)),
                },
            }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QtWidgets.QMessageBox.information(
                self, "Save", f"Calibration saved to {path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save", str(e))

    def _load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Calibration", "calibration.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load", str(e))
            return

        try:
            for i, p in enumerate(self.panels, start=1):
                key = f"camera_{i}"
                if key not in data:
                    continue
                entry = data[key]

                # Punkte
                pts = np.array(entry.get("points_img", []), dtype=np.float32)
                if pts.shape == (4, 2):
                    p.view.set_points(pts)

                # Intrinsics / Undistort
                intr = entry.get("intrinsics", {})
                p.k1.setValue(float(intr.get("k1", 0.0)))
                p.k2.setValue(float(intr.get("k2", 0.0)))
                p.cb_undist.setChecked(bool(intr.get("enabled", False)))

                # Overlay-Optik
                ov = entry.get("overlay", {})
                if "alpha" in ov:
                    p.view.alpha = float(ov["alpha"])
                if "wedge_angle_deg" in ov:
                    p.view.set_wedge_template(float(ov["wedge_angle_deg"]))

                # Homographien
                H_w = entry.get("H_world2img")
                if H_w is not None:
                    try:
                        H_w = np.array(H_w, dtype=np.float64)
                        p.view.H_world2img = H_w
                        p.view.H_tplPx2img = templatePx_to_image_H(H_w)
                    except Exception:
                        p.view.H_world2img = None
                        p.view.H_tplPx2img = None
                else:
                    # falls nicht in Datei: aus Punkten/Frame berechnen (ohne Refinement)
                    p.view.set_snapshot_from_current_frame()
                    p.view.recompute_homography(refine=False)

                p.view.update()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load", str(e))

    # -------- Save to Agent (/calibration) --------
    def _save_agent(self):
        Hs_overlay = []
        Hs_world = []
        intr_list = []

        # vor dem Sammeln sicherstellen, dass Hs gefüllt sind
        for p in self.panels:
            p.view.set_snapshot_from_current_frame()
            p.view.recompute_homography(refine=False)

        for p in self.panels:
            H_tpl = p.current_overlay_h()
            H_w = p.current_world_h()
            if H_tpl is not None:
                Hs_overlay.append(H_tpl.tolist())
            if H_w is not None:
                Hs_world.append(H_w.tolist())
            intr_list.append(
                {
                    "k1": float(p.k1.value()),
                    "k2": float(p.k2.value()),
                    "enabled": bool(p.cb_undist.isChecked()),
                }
            )

        if not Hs_world:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Keine gültigen Homographien vorhanden. Punkte setzen oder Auto nutzen.",
            )
            return

        cfg = fetch_calibration()
        if not isinstance(cfg, dict):
            cfg = {}

        cfg["boardDiameterMm"] = 451.0
        cfg["rings"] = REL
        res = self.cmb_res.currentData()
        fps = int(self.cmb_fps.currentData())
        cfg["capture"] = {"resolution": [int(res[0]), int(res[1])], "fps": fps}
        cfg["homographies_world2img"] = Hs_world
        cfg["homographies_overlay_tplPx2img"] = Hs_overlay
        cfg["intrinsics"] = intr_list

        try:
            save_calibration(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save", f"Fehler beim Speichern beim Agent:\n{e}"
            )
            return

        QtWidgets.QMessageBox.information(
            self, "Save", "Kalibrierung beim Agent gespeichert."
        )

    # -------- Auto-Last-Session (AppData) --------
    def _save_last_session(self):
        """Speichert UI- und Kalibrierzustand in AppData."""
        data = {
            "capture": {
                "resolution": list(map(int, self.cmb_res.currentData())),
                "fps": int(self.cmb_fps.currentData()),
            },
            "panels": [],
        }
        for p in self.panels:
            # defensiv Homographien (ohne Refinement)
            p.view.set_snapshot_from_current_frame()
            p.view.recompute_homography(refine=False)

            H_tpl = p.current_overlay_h()
            H_w = p.current_world_h()
            cam_id = p.cmb.currentData()  # kann None sein
            panel = {
                "camera": None if cam_id is None else int(cam_id),
                "intrinsics": {
                    "k1": float(p.k1.value()),
                    "k2": float(p.k2.value()),
                    "enabled": bool(p.cb_undist.isChecked()),
                },
                "points_img": p.current_points(),
                "H_overlay_tplPx2img": (H_tpl.tolist() if H_tpl is not None else None),
                "H_world2img": (H_w.tolist() if H_w is not None else None),
                "overlay": {
                    "alpha": float(p.view.alpha),
                    "wedge_angle_deg": float(getattr(p.view, "_wedge_angle_deg", 90.0)),
                },
            }
            data["panels"].append(panel)

        path = self._appdata_calib_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print("WARN save last session:", e)

    def _load_last_session(self):
        """Lädt UI- und Kalibrierzustand aus AppData, falls vorhanden."""
        path = self._appdata_calib_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("WARN load last session:", e)
            return

        # Capture (setzt Kombos → triggert _apply_globals)
        cap = data.get("capture", {})
        res = tuple(cap.get("resolution", (1280, 720)))
        fps = int(cap.get("fps", 30))

        # Auflösung wählen
        for i in range(self.cmb_res.count()):
            if self.cmb_res.itemData(i) == res:
                self.cmb_res.setCurrentIndex(i)
                break
        # FPS wählen
        for i in range(self.cmb_fps.count()):
            if int(self.cmb_fps.itemData(i)) == fps:
                self.cmb_fps.setCurrentIndex(i)
                break

        # Panels
        panels = data.get("panels", [])
        for idx, panel in enumerate(panels):
            if idx >= len(self.panels):
                break
            p = self.panels[idx]

            # Kameraauswahl (falls noch vorhanden)
            cam_id = panel.get("camera", None)
            if cam_id is not None:
                for i in range(p.cmb.count()):
                    if p.cmb.itemData(i) == cam_id:
                        p.cmb.setCurrentIndex(i)
                        break

            # Intrinsics/Undistort
            intr = panel.get("intrinsics", {})
            p.k1.setValue(float(intr.get("k1", 0.0)))
            p.k2.setValue(float(intr.get("k2", 0.0)))
            p.cb_undist.setChecked(bool(intr.get("enabled", False)))

            # Punkte
            pts = panel.get("points_img")
            if pts and isinstance(pts, list) and len(pts) == 4:
                p.view.set_points(np.array(pts, dtype=np.float32))

            # Overlay
            ov = panel.get("overlay", {})
            if "alpha" in ov:
                p.view.alpha = float(ov["alpha"])
            if "wedge_angle_deg" in ov:
                p.view.set_wedge_template(float(ov["wedge_angle_deg"]))

            # H direkt einsetzen (kein Frame nötig)
            H_w = panel.get("H_world2img")
            if H_w is not None:
                try:
                    H_w = np.array(H_w, dtype=np.float64)
                    p.view.H_world2img = H_w
                    p.view.H_tplPx2img = templatePx_to_image_H(H_w)
                except Exception:
                    p.view.H_world2img = None
                    p.view.H_tplPx2img = None

            p.view.update()

    # -------- Window close: auto-save --------
    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self._save_last_session()
        finally:
            super().closeEvent(e)


def open_triple_calibration(parent=None):
    dlg = TripleCalibration(parent)
    dlg.exec()
