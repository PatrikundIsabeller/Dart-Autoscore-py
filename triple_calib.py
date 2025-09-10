# triple_calib.py – 3-Kamera-Kalibrierung (PyQt6) mit Drag-Points + Save zu /calibration
from __future__ import annotations
import os, json
from typing import List, Tuple, Optional

import cv2
import numpy as np
import requests
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFrame

AGENT_URL = os.environ.get("TRIPLEONE_AGENT", "http://127.0.0.1:4700")

DEFAULT_RINGS = {
    "double_outer": 0.995,
    "double_inner": 0.940,
    "triple_outer": 0.620,
    "triple_inner": 0.540,
    "bull_outer": 0.060,
    "bull_inner": 0.015,
}

# ---------- Auto-Detect & Auto-Homography ----------

def _detect_outer_ellipse(frame: np.ndarray) -> Optional[Tuple[Tuple[float,float], Tuple[float,float], float]]:
    """
    Liefert (center(x,y), axes(major,minor), angle_deg) der größten 'ring-ähnlichen' Ellipse.
    Robust genug für Soft-/Steeldart mit Rot/Grün; fällt zurück auf Kanten, wenn Farben schwach.
    """
    if frame is None or frame.size == 0:
        return None

    work = frame.copy()
    # 1) leichtes entrauschen / Kontrast
    work = cv2.GaussianBlur(work, (5,5), 0)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    # 2) Rot maskieren (2 Bereiche in HSV)
    low1 = np.array([0, 80, 60]);   high1 = np.array([12, 255, 255])
    low2 = np.array([170, 80, 60]); high2 = np.array([179, 255, 255])
    mask_r = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)

    # 3) Kanten als Fallback kombinieren
    g = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    edges = cv2.Canny(g, 60, 140)
    mask = cv2.bitwise_or(mask_r, edges)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None; best_score = -1.0
    H, W = mask.shape[:2]
    area_img = H * W

    for c in cnts:
        if len(c) < 20: 
            continue
        area = cv2.contourArea(c)
        if area < 0.01 * area_img:
            continue
        try:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)   # major, minor
        except cv2.error:
            continue
        # Score: groß + möglichst rund
        ar = min(MA, ma) / max(MA, ma)  # 1 = rund
        score = area * (0.5 + 0.5*ar)
        if score > best_score:
            best_score = score
            best = ((cx, cy), (MA, ma), ang)

    return best  # oder None


def _ellipse_rad_at_angle(a: float, b: float, phi_img: float, theta_deg: float) -> float:
    """
    Abstand vom Ellipsenzentrum zur Ellipsenkante entlang 'phi_img' (Bildwinkel),
    gegeben Halbachsen a,b und Ellipsendrehung theta (Grad).
    Formel: r = 1 / sqrt((cos(φ')^2 / a^2) + (sin(φ')^2 / b^2)), wobei φ' = φ - θ.
    """
    theta = np.deg2rad(theta_deg)
    phi_p = phi_img - theta
    c, s = np.cos(phi_p), np.sin(phi_p)
    den = (c*c)/(a*a) + (s*s)/(b*b)
    if den <= 1e-12:
        return 0.0
    return 1.0 / np.sqrt(den)


def _auto_rotation_offset(gray: np.ndarray, center: Tuple[float,float], radius_px: float) -> float:
    """
    Schätzt den Rotationswinkel (Grad), bei dem ein 20-Segmente-Template maximal korreliert.
    Wir sampeln eine Ring-Intensität (zwischen Triple und Double), bauen ein 360-Bin-Profil
    und korrelieren mit einer 20er-Kammfunktion.
    """
    h, w = gray.shape[:2]
    cx, cy = center
    r1 = radius_px * 0.62    # ungefähr triple_outer
    r2 = radius_px * 0.95    # ungefähr double_outer
    r = (r1 + r2) / 2.0

    # 360 Samples entlang des Rings
    prof = np.zeros(360, dtype=np.float32)
    for deg in range(360):
        phi = np.deg2rad(deg)
        x = int(cx + r * np.cos(phi))
        y = int(cy + r * np.sin(phi))
        if 0 <= x < w and 0 <= y < h:
            prof[deg] = gray[y, x]

    # Hochpass / normalisieren
    prof = (prof - np.median(prof))
    prof /= (np.std(prof) + 1e-6)

    # 20er-Kamm: Peaks alle 18°
    # Wir nehmen einfach einen Sinus auf 20/2 = 10 Zyklen (approx. Kamm) – stabiler als harte Diracs
    t = np.arange(360, dtype=np.float32) * np.pi / 180.0
    template = np.cos(10 * 2 * np.pi * t / (2*np.pi))  # 10 Perioden in 360°
    # FFT-Korrelation
    fft_corr = np.fft.ifft(np.fft.fft(prof) * np.conj(np.fft.fft(template))).real
    best_deg = int(np.argmax(fft_corr) % 360)

    # Wir wollen, dass Segment 20 "oben" (−90°) liegt → Offset umrechnen:
    # Unser Grid startet bei 90° (oben) → Ausrichtung = (90 - best_deg)
    rot = 90 - best_deg
    # Normalisieren in [-180, 180]
    while rot > 180: rot -= 360
    while rot < -180: rot += 360
    return float(rot)


def auto_calibrate_from_frame(frame: np.ndarray) -> Optional[dict]:
    """
    Findet äußere Ellipse, schätzt Rotation (20 nach oben),
    setzt 4 Punkte passend zur Rotation auf die Kardinalpunkte
    (TopLeft, TopRight, BottomRight, BottomLeft) und berechnet H.
    Rückgabe: {"points": [(x,y)*4], "H": 3x3, "rotation_deg": float}
    """
    det = _detect_outer_ellipse(frame)
    if det is None:
        return None
    (cx, cy), (MA, ma), ang = det
    a, b = MA / 2.0, ma / 2.0  # Halbachsen

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rot_est = _auto_rotation_offset(gray, (cx, cy), max(a, b))  # +rot => Grid gegen Uhr, 20 nach oben

    # --- Winkel (Bildkoordinaten) für die vier Ecken des 600x600-Boards:
    # Mapping (src -> dst):
    #   (0,0)     -> Top-Left     -> 135° - rot
    #   (600,0)   -> Top-Right    ->  45° - rot
    #   (600,600) -> Bottom-Right -> -45° (oder 315°) - rot
    #   (0,600)   -> Bottom-Left  -> 225° - rot
    def corner_point(angle_deg: float) -> List[float]:
        phi = np.deg2rad(angle_deg)
        r = _ellipse_rad_at_angle(a, b, phi, ang)
        x = cx + r * np.cos(phi)
        y = cy + r * np.sin(phi)
        return [float(x), float(y)]

    angles_deg = [
        135.0 - rot_est,   # TL
        45.0  - rot_est,   # TR
        -45.0 - rot_est,   # BR (== 315° - rot)
        225.0 - rot_est    # BL
    ]
    dst = [corner_point(d) for d in angles_deg]
    dst_np = np.array(dst, dtype=np.float32)

    src = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)

    H, ok = cv2.findHomography(src, dst_np, 0)
    if H is None:
        return None

    return {"points": dst, "H": H, "rotation_deg": float(rot_est)}



# ---------------- HTTP ----------------
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


# -------------- Camera utils --------------
def open_capture(cam_id: int, res=(1280, 720), fps: int = 30):
    """Versucht MSMF → DSHOW → ANY. Gibt geöffnetes cap oder None zurück."""
    backends = [
        getattr(cv2, "CAP_MSMF", 1400),
        getattr(cv2, "CAP_DSHOW", 700),
        getattr(cv2, "CAP_ANY", 0),
    ]
    for be in backends:
        try:
            cap = cv2.VideoCapture(cam_id, be)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            cap.set(cv2.CAP_PROP_FPS, fps)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            continue
    return None


def enumerate_cameras(max_idx: int = 10) -> list[int]:
    found = []
    for i in range(max_idx):
        cap = open_capture(i, (320, 240), 15)
        if cap:
            found.append(i)
            cap.release()
    return found


# -------------- Image utils --------------
def cvimg_to_qpix(img: np.ndarray) -> QtGui.QPixmap:
    if img is None or img.size == 0:
        return QtGui.QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QPixmap.fromImage(
        QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
    )


def draw_dartboard_grid(size: int = 600, rings: dict | None = None) -> np.ndarray:
    """Board-Template in 600x600 mit korrekten Ringradien."""
    if rings is None:
        rings = DEFAULT_RINGS

    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    R = size // 2

    white = (255, 255, 255)
    red   = (0, 0, 255)
    green = (0, 255, 0)
    yellow= (0, 255, 255)
    blue  = (255, 0, 0)

    # Double-Ring
    cv2.circle(img, (cx, cy), int(R * rings["double_outer"]), red,   2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * rings["double_inner"]), red,   2, cv2.LINE_AA)

    # Triple-Ring
    cv2.circle(img, (cx, cy), int(R * rings["triple_outer"]), green, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * rings["triple_inner"]), green, 2, cv2.LINE_AA)

    # Bull / Bullseye
    cv2.circle(img, (cx, cy), int(R * rings["bull_inner"]),   blue,  -1, cv2.LINE_AA)   # inner bull
    cv2.circle(img, (cx, cy), int(R * rings["bull_outer"]),   yellow,2, cv2.LINE_AA)   # outer bull

    # Segmente + Zahlen (20-Start oben; 18° pro Segment)
    nums = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    for i in range(20):
        ang = np.deg2rad(90 - i * 18)  # 90° = oben
        r0 = R * rings["bull_outer"]
        r1 = R * rings["double_outer"]
        x0 = int(cx + r0 * np.cos(ang)); y0 = int(cy - r0 * np.sin(ang))
        x1 = int(cx + r1 * np.cos(ang)); y1 = int(cy - r1 * np.sin(ang))
        cv2.line(img, (x0, y0), (x1, y1), white, 1, cv2.LINE_AA)

        # Zahlen zwischen double_outer und Boardrand
        txt = str(nums[i])
        rad = R * 1.02
        tx = int(cx + rad * np.cos(ang)); ty = int(cy - rad * np.sin(ang))
        sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(img, txt, (tx - sz[0] // 2, ty + sz[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2, cv2.LINE_AA)
    return img



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


# -------------- Drag widget (single feed) --------------
class DragBoard(QtWidgets.QLabel):
    """Liveframe + warped Grid + 4 draggable Punkte (TL,TR,BR,BL) mit Feinjustage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setStyleSheet("QLabel{border:1px solid #404040;border-radius:8px;background:#111;color:#fff;}")

        # --- Zustand -------------------------------------------------
        self.frame: Optional[np.ndarray] = None
        self.points = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], dtype=np.float32)

        # Homographie-Matrix (wird in homography() berechnet)
        self.M: Optional[np.ndarray] = None

        # Polygon für das rote Tortenstück (Bild-Koordinaten) – None = aus
        self._wedge_poly: list[tuple[float, float]] | None = None


        # State
        self.frame: Optional[np.ndarray] = None
        self.points = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], dtype=np.float32)
        self.M: Optional[np.ndarray] = None
        self._wedge_poly = None  # Liste von (x,y) Bildkoordinaten
        self._drag_idx = -1
        self._active_idx = 0  # Tastaturfokus: 0..3

        # Overlay/Board-Parameter (werden vom Panel gesetzt)
        self.alpha = 0.65           # 0..1
        self.board_scale = 1.000    # 0.95..1.05
        self.board_rotate = 0.0     # Grad (-5..+5)

    def set_wedge_poly(self, poly: Optional[list[tuple[float, float]]]):
        """Polygon des 20er-Sektors in Bildkoordinaten setzen (oder None zum Ausblenden)."""
        self._wedge_poly = poly
        self.update()

    # ---------- public setter ----------
    def set_alpha(self, a: float):
        self.alpha = max(0.0, min(1.0, a)); self.update()

    def set_board_scale(self, s: float):
        self.board_scale = max(0.90, min(1.10, s)); self.update()

    def set_board_rotate(self, deg: float):
        self.board_rotate = max(-15.0, min(15.0, deg)); self.update()

    # ---------- core ----------
    def set_frame(self, frame: Optional[np.ndarray]):
        self.frame = frame
        self.update()

    def _src_square(self) -> np.ndarray:
        """600x600-Quad nach Scale/Rotation um das Zentrum transformieren."""
        src = np.array([[0,0],[600,0],[600,600],[0,600]], dtype=np.float32)
        c = np.array([300.0, 300.0], dtype=np.float32)
        v = src - c
        rad = np.deg2rad(self.board_rotate)
        R = np.array([[np.cos(rad), -np.sin(rad)],
                      [np.sin(rad),  np.cos(rad)]], dtype=np.float32)
        v = (R @ (v.T)).T * self.board_scale
        return v + c

    def homography(self) -> Optional[np.ndarray]:
        if self.points.shape != (4, 2):
            return None
        src = self._src_square()
        dst = self.points.astype(np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst)
        return self.M
    
    def set_wedge_poly(self, poly: Optional[list[tuple[float, float]]]):
        """Polygon des 20er-Sektors in Bildkoordinaten setzen (oder None zum Ausblenden)."""
        self._wedge_poly = poly
        self.update()


    # ---------- painting ----------
    def _paint_pixmap(self, p: QtGui.QPainter, pm: QtGui.QPixmap):
        rect = self.rect()
        ps = pm.size()
        scale = min(rect.width() / ps.width(), rect.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (rect.width() - dw) / 2.0, (rect.height() - dh) / 2.0
        p.drawPixmap(QtCore.QRectF(ox, oy, dw, dh), pm, QtCore.QRectF(0, 0, ps.width(), ps.height()))
        return scale, ox, oy, ps

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        if self.frame is None:
            p.setPen(QtGui.QPen(QtGui.QColor("#aaa")))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No signal")
            return

        pm = cvimg_to_qpix(self.frame)
        scale, ox, oy, ps = self._paint_pixmap(p, pm)

        # Overlay: warped Grid
        if self.homography() is not None:
            h, w = self.frame.shape[:2]
            warped = cv2.warpPerspective(GRID_600, self.M, (w, h))
            wpm = cvimg_to_qpix(warped)
            p.setOpacity(0.7)
            self._paint_pixmap(p, wpm)
            p.setOpacity(1.0)

        # --- rotes Tortenstück (falls gesetzt) ---  <<< DIESER NEUE BLOCK
        if self._wedge_poly and self.frame is not None:
            fw, fh = self.frame.shape[1], self.frame.shape[0]
            path = QtGui.QPainterPath()
            first = True
            for (x_img, y_img) in self._wedge_poly:
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

        # Punkte (dein bestehender Code)
        p.setPen(QtGui.QPen(QtGui.QColor("#22c55e"), 3))
        p.setBrush(QtGui.QColor(34, 197, 94, 180))
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        for idx, (x_img, y_img) in enumerate(self.points):
            x = ox + (x_img / fw) * (ps.width() * scale)
            y = oy + (y_img / fh) * (ps.height() * scale)
            p.drawEllipse(QtCore.QPointF(float(x), float(y)), 6, 6)
            p.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
            p.drawText(QtCore.QPointF(float(x) + 8.0, float(y) - 8.0), str(idx + 1))
            p.setPen(QtGui.QPen(QtGui.QColor("#22c55e"), 3))

    # ---------- mouse/keyboard ----------
    def _img_coords_from_mouse(self, e: QtGui.QMouseEvent):
        pm = cvimg_to_qpix(self.frame)
        rect = self.rect(); ps = pm.size()
        scale = min(rect.width() / ps.width(), rect.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (rect.width() - dw) / 2.0, (rect.height() - dh) / 2.0
        pos = e.position()
        x = (pos.x() - ox) / scale; y = (pos.y() - oy) / scale
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        x_img = max(0, min(fw - 1, x * (fw / ps.width())))
        y_img = max(0, min(fh - 1, y * (fh / ps.height())))
        return x_img, y_img

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self.frame is None: return
        xi, yi = self._img_coords_from_mouse(e)
        d = np.linalg.norm(self.points - np.array([xi, yi]), axis=1)
        self._drag_idx = int(np.argmin(d)) if d.size else -1
        if self._drag_idx >= 0: self._active_idx = self._drag_idx

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.frame is None or self._drag_idx < 0: return
        xi, yi = self._img_coords_from_mouse(e)
        self.points[self._drag_idx] = [xi, yi]
        self.update()

    def mouseReleaseEvent(self, _e: QtGui.QMouseEvent):
        self._drag_idx = -1

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        # 1..4 = Punktwahl, Pfeile = nudgen (Shift=5px, Ctrl=20px)
        k = e.key()
        if k in (QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3, QtCore.Qt.Key.Key_4):
            self._active_idx = int(k - QtCore.Qt.Key.Key_1)
            self.update(); return
        step = 1.0
        if e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier: step = 5.0
        if e.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier: step = 20.0
        if k == QtCore.Qt.Key.Key_Left:  self.points[self._active_idx, 0] -= step
        if k == QtCore.Qt.Key.Key_Right: self.points[self._active_idx, 0] += step
        if k == QtCore.Qt.Key.Key_Up:    self.points[self._active_idx, 1] -= step
        if k == QtCore.Qt.Key.Key_Down:  self.points[self._active_idx, 1] += step
        # Q/E rotieren, W/S skalieren minimal
        if k == QtCore.Qt.Key.Key_Q: self.board_rotate -= 0.2
        if k == QtCore.Qt.Key.Key_E: self.board_rotate += 0.2
        if k == QtCore.Qt.Key.Key_W: self.board_scale  += 0.001
        if k == QtCore.Qt.Key.Key_S: self.board_scale  -= 0.001
        self.update()



# -------------- Kamera-Panel (eine Kachel) --------------
SECTOR_SEQ = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

def _fit_h_from_ellipse(center, axes, tilt_deg, rot_final_deg) -> Optional[np.ndarray]:
    """Viele (src,dst)-Paare erzeugen und Homography robust schätzen (LMEDS)."""
    (cx, cy), (MA, ma) = center, axes
    a, b = MA/2.0, ma/2.0
    ang = tilt_deg

    src_pts, dst_pts = [], []
    R = 300.0
    template_radii = [R, R*0.95, R*0.66, R*0.60, R*0.12, R*0.06]  # außen..bull
    for t in range(16):
        theta = 2.0*np.pi * t/16.0
        for r in template_radii:
            sx = 300.0 + r*np.cos(theta)
            sy = 300.0 + r*np.sin(theta)

            phi_deg = np.rad2deg(theta) - rot_final_deg
            phi = np.deg2rad(phi_deg)

            rout = _ellipse_rad_at_angle(a, b, phi, ang)  # Radius außen
            rdst = rout * (r / R)

            dstx = cx + rdst*np.cos(phi)
            dsty = cy + rdst*np.sin(phi)

            src_pts.append([sx, sy])
            dst_pts.append([dstx, dsty])

    src_np = np.array(src_pts, dtype=np.float32)
    dst_np = np.array(dst_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src_np, dst_np, method=cv2.LMEDS)
    return H


def _make_wedge_poly(center, axes, tilt_deg, rot_final_deg, span_deg=18.0, frac=0.80) -> list[tuple[float,float]]:
    """
    Baut ein Polygon für das rote Tortenstück (zentral -> Bogen -> zentral).
    span_deg: 18° für eine Sektorspalte; frac: Anteil des Außenradius.
    """
    (cx, cy), (MA, ma) = center, axes
    a, b = MA/2.0, ma/2.0
    ang = tilt_deg

    mid = np.deg2rad(0.0 - rot_final_deg)    # Richtung "oben" nach Rotation
    half = np.deg2rad(span_deg/2.0)
    start = mid - half
    end   = mid + half

    # Bogenpunkte entlang des äußeren Radius (leicht innen)
    arc = []
    for t in np.linspace(start, end, 20):
        rout = _ellipse_rad_at_angle(a, b, t, ang) * frac
        x = cx + rout*np.cos(t)
        y = cy + rout*np.sin(t)
        arc.append((float(x), float(y)))

    return [(float(cx), float(cy))] + arc + [(float(cx), float(cy))]


# ----------------------- CamPanel -----------------------
class CamPanel(QtWidgets.QFrame):
    """
    Eine Kamera-Kachel:
      - Kamera wählen
      - Undistort (k1/k2)
      - scale / rot
      - Auto: findet Board-Kreis + richtet 20°-Wedge aus
    """
    def __init__(self, cam_id_guess: int):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # Ansicht
        self.view = DragBoard()

        # Kameraauswahl (Items füllt TripleCalibration)
        self.cmb = QtWidgets.QComboBox()
        self.cmb.addItem("Select camera", None)
        self.cmb.currentIndexChanged.connect(self._on_select)

        # Entzerrung (grob)
        self.cb_undist = QtWidgets.QCheckBox("Undistort")
        self.k1 = QtWidgets.QDoubleSpinBox(); self.k1.setRange(-0.50, 0.50); self.k1.setDecimals(3); self.k1.setSingleStep(0.005); self.k1.setValue(0.000); self.k1.setPrefix("k1 ")
        self.k2 = QtWidgets.QDoubleSpinBox(); self.k2.setRange(-0.50, 0.50); self.k2.setDecimals(3); self.k2.setSingleStep(0.005); self.k2.setValue(0.000); self.k2.setPrefix("k2 ")

        # Overlays: Scale/Rotation
        self.scale = QtWidgets.QDoubleSpinBox(); self.scale.setRange(0.80, 1.40); self.scale.setDecimals(3); self.scale.setSingleStep(0.002); self.scale.setValue(1.000); self.scale.setPrefix("scale ")
        self.rot   = QtWidgets.QDoubleSpinBox(); self.rot.setRange(-180.0, 180.0); self.rot.setDecimals(1); self.rot.setSingleStep(0.5); self.rot.setValue(0.0); self.rot.setPrefix("rot ")

        # Auto-Knopf
        self.btn_auto = QtWidgets.QPushButton("Auto")

        # UI-Leiste unten
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.cmb, 1)
        bar.addWidget(self.cb_undist, 0)
        bar.addWidget(self.k1, 0)
        bar.addWidget(self.k2, 0)
        bar.addWidget(self.scale, 0)
        bar.addWidget(self.rot, 0)
        bar.addWidget(self.btn_auto, 0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)

        # Verbindungen
        self.cb_undist.stateChanged.connect(lambda *_: self.view.update())
        self.k1.valueChanged.connect(lambda *_: self.view.update())
        self.k2.valueChanged.connect(lambda *_: self.view.update())
        self.scale.valueChanged.connect(lambda *_: self._apply_transform())
        self.rot.valueChanged.connect(lambda *_: self._apply_transform())
        self.btn_auto.clicked.connect(self.auto_calibrate)

        # Laufzeitstate
        self.cap: Optional[cv2.VideoCapture] = None
        self.res = (1280, 720)
        self.fps = 30

        # Ergebnis der Auto-Detektion (für Live-Nachführung bei rot/scale)
        self._last_center: Optional[tuple[float, float]] = None
        self._last_radius: Optional[float] = None

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
            QtWidgets.QMessageBox.warning(self, "Camera", f"Cannot open camera {cam_id} on any backend.")
            return
        self.cap = cap
        self._last_center = None
        self._last_radius = None
        self.view.set_wedge_poly(None)  # Tortenstück zurücksetzen

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.view.set_frame(None)

    def tick(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.view.set_frame(None)
            return
        if self.cb_undist.isChecked():
            frame = self._undistort_simple(frame, float(self.k1.value()), float(self.k2.value()))
        self.view.set_frame(frame.copy())

    def set_globals(self, res: tuple[int, int], fps: int):
        self.res = res
        self.fps = fps
        if self.cap:
            idx = self.cmb.currentData()
            if idx is not None:
                self.start(int(idx))

    def current_h(self) -> Optional[np.ndarray]:
        return self.view.homography()

    def current_points(self) -> list[list[float]]:
        return self.view.points.tolist()

    # ---------- Auto-Kalibrierung ----------
    def auto_calibrate(self):
        """
        1) Frame holen
        2) größten Kreis (Board) mit HoughCircles finden -> center (cx,cy), radius r
        3) dominanten Radial-Strahl via HoughLines (durch's Zentrum) -> Winkel -> rot
        4) aus center/scale/rot vier Zielpunkte berechnen -> view.points setzen
        5) 20er-Wedge als Polygon zeichnen
        """
        if not self.cap:
            # versuchen, aus Auswahl zu starten
            cam = self.cmb.currentData()
            if cam is None:
                QtWidgets.QMessageBox.information(self, "Auto", "Bitte zuerst eine Kamera wählen.")
                return
            self.start(int(cam))
            if not self.cap:
                return

        # stabiles Einzelbild holen
        ok, frame = self.cap.read()
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Auto", "Kein Kamerabild verfügbar.")
            return

        work = frame.copy()
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

        # 2) Kreis suchen (rel. großzügige Parameter)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rows/4,
            param1=100, param2=30, minRadius=int(rows*0.20), maxRadius=int(rows*0.50)
        )

        if circles is None:
            QtWidgets.QMessageBox.warning(self, "Auto", "Kein Board-Kreis gefunden. (Sorge für gutes Licht / kontrastreiche Ringe.)")
            return

        circles = np.uint16(np.around(circles))
        # größten nehmen
        cxs, cys, rs = circles[0,:,0], circles[0,:,1], circles[0,:,2]
        max_idx = int(np.argmax(rs))
        cx, cy, r = float(cxs[max_idx]), float(cys[max_idx]), float(rs[max_idx])

        # 3) Winkel des dominanten Radial-Strahls schätzen
        edges = cv2.Canny(gray, 80, 160)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
        rot_deg = 0.0
        if lines is not None:
            # Linie mit minimaler Distanz zum Zentrum wählen
            best = None
            best_d = 1e9
            for rho_theta in lines[:,0,:]:
                rho, theta = float(rho_theta[0]), float(rho_theta[1])
                # Distanz Punkt (cx,cy) zur Linie (rho,theta):
                d = abs(cx*np.cos(theta) + cy*np.sin(theta) - rho)
                if d < best_d:
                    best_d, best = d, (rho, theta)
            if best is not None:
                _, theta = best
                # Linienrichtung = theta + 90°; wir wollen, dass diese "nach oben" (90°) zeigt
                angle_line = np.degrees(theta) + 90.0
                # zu [-180,180] normalisieren
                while angle_line > 180: angle_line -= 360
                while angle_line < -180: angle_line += 360
                rot_deg = 90.0 - angle_line  # Korrektur
        # UI aktualisieren (User darf feinjustieren)
        self.rot.blockSignals(True)
        self.rot.setValue(rot_deg)
        self.rot.blockSignals(False)

        # merken für Folgeregelung
        self._last_center = (cx, cy)
        self._last_radius = r

        # 4/5) Punkte + Wedge anhand aktueller Regler anwenden
        self._apply_transform()

    # ---------- Geometrie anwenden ----------
    def _apply_transform(self):
        """Erzeuge points (Ziel-Quadrilateral) und Wedge aus center/r/scale/rot."""
        if self._last_center is None or self._last_radius is None:
            return
        cx, cy = self._last_center
        r = float(self._last_radius)
        s = float(self.scale.value())
        rot_deg = float(self.rot.value())
        rot_rad = np.deg2rad(rot_deg)

        # Quadratecken um (cx,cy) mit Kantenlänge 2*s*r
        half = s * r
        corners = np.array([
            [-half, -half], [ half, -half], [ half,  half], [-half,  half]
        ], dtype=np.float32)

        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad),  np.cos(rot_rad)]], dtype=np.float32)
        dst = (corners @ R.T) + np.array([cx, cy], dtype=np.float32)

        self.view.points = dst.astype(np.float32)  # DragBoard holt daraus self.M
        # Wedge: 20er-Sektor (±9° um "oben")
        wedge = self._make_wedge_poly((cx, cy), r_out=r*0.98, r_in=r*0.10, angle_up_deg=90.0 + rot_deg, spread_deg=18.0)
        self.view.set_wedge_poly(wedge)
        self.view.update()

    # ---------- Helper ----------
    @staticmethod
    def _undistort_simple(img: np.ndarray, k1: float, k2: float) -> np.ndarray:
        """Simple radiale Entzerrung (ohne Kameramatrix). Reicht zum Ausrichten."""
        if abs(k1) < 1e-8 and abs(k2) < 1e-8:
            return img
        h, w = img.shape[:2]
        cx, cy = w/2.0, h/2.0
        yy, xx = np.indices((h, w), dtype=np.float32)
        x = (xx - cx) / cx
        y = (yy - cy) / cy
        r2 = x*x + y*y
        factor = 1 + k1*r2 + k2*(r2*r2)
        x_u = x * factor
        y_u = y * factor
        map_x = (x_u * cx + cx).astype(np.float32)
        map_y = (y_u * cy + cy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _make_wedge_poly(center: tuple[float, float], r_out: float, r_in: float,
                         angle_up_deg: float, spread_deg: float = 18.0,
                         steps: int = 12) -> list[tuple[float, float]]:
        """
        Erzeugt ein Ring-Sektor-Polygon (für das 20er-Tortenstück):
        - angle_up_deg: Richtung "oben" in Bildkoordinaten
        - spread_deg: Sektorbreite (Darts = 18°)
        """
        cx, cy = center
        a0 = np.deg2rad(angle_up_deg - spread_deg/2.0)
        a1 = np.deg2rad(angle_up_deg + spread_deg/2.0)
        outer = []
        inner = []
        for t in np.linspace(a0, a1, steps):
            outer.append((cx + r_out*np.cos(t), cy - r_out*np.sin(t)))
        for t in np.linspace(a1, a0, steps):
            inner.append((cx + r_in*np.cos(t), cy - r_in*np.sin(t)))
        return outer + inner


# -------------- Main-Dialog --------------
class TripleCalibration(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure cameras")
        self.resize(1280, 760)

        title = QtWidgets.QLabel("Configure cameras")
        title.setStyleSheet("font-size:24px;font-weight:700;margin:8px 0 16px 0;")

        self.panels = [CamPanel(0), CamPanel(1), CamPanel(2)]

        # nur vorhandene Kameras ins Dropdown
        avail = enumerate_cameras(10)  # z. B. [0]
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

        # globals
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

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(title)
        root.addLayout(row)
        root.addSpacing(8)
        root.addLayout(grid)

        # wiring
        self.cmb_res.currentIndexChanged.connect(self._apply_globals)
        self.cmb_fps.currentIndexChanged.connect(self._apply_globals)
        self.btn_save_agent.clicked.connect(self._save_agent)
        self.btn_save_file.clicked.connect(self._save_file)
        self.btn_load_file.clicked.connect(self._load_file)

        # timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._apply_globals()  # kein Autostart – Kamera erst nach Auswahl

    def _apply_globals(self):
        res = self.cmb_res.currentData()
        fps = int(self.cmb_fps.currentData())
        for p in self.panels:
            p.set_globals(res, fps)

    def _tick(self):
        for p in self.panels:
            p.tick()

    # --- Save/Load (Fiverr-JSON kompatibel) ---
    def _save_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration", "calibration.json", "JSON (*.json)"
        )
        if not path:
            return
        data = {}
        for i, p in enumerate(self.panels, start=1):
            H = p.current_h()
            if H is None:
                continue
            data[f"camera_{i}"] = {
                "points": p.current_points(),
                "homography_matrix": H.tolist(),
                "intrinsics": {"k1": float(p.k1.value()), "k2": float(p.k2.value())},
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        QtWidgets.QMessageBox.information(self, "Save", f"Calibration saved to {path}")

    def _load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Calibration", "calibration.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for i, p in enumerate(self.panels, start=1):
                key = f"camera_{i}"
                if key in data:
                    # Punkte übernehmen
                    pts = np.array(data[key]["points"], dtype=np.float32)
                    p.view.points = pts

                    # ➜ Intrinsics (k1/k2) übernehmen
                    intr = data[key].get("intrinsics")
                    if intr:
                        p.k1.setValue(float(intr.get("k1", 0.0)))
                        p.k2.setValue(float(intr.get("k2", 0.0)))
                        p.cb_undist.setChecked(
                            abs(p.k1.value()) > 1e-6 or abs(p.k2.value()) > 1e-6
                        )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load", str(e))

    # --- Save to Agent (/calibration) ---
    def _save_agent(self):
        # 1) Homographien einsammeln
        Hs = []
        intr_list = []
        for p in self.panels:
            H = p.current_h()
            if H is not None:
                Hs.append(H.tolist())

            intr_list.append(
                {
                    "k1": float(p.k1.value()),
                    "k2": float(p.k2.value()),
                    "enabled": bool(p.cb_undist.isChecked()),
                }
            )

        if not Hs:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Keine Homographien vorhanden (Punkte ausrichten und/oder Kamera wählen).",
            )
            return

        # 2) Bestehende Config holen und Defaults setzen
        cfg = fetch_calibration()
        if not isinstance(cfg, dict):
            cfg = {}

        cfg.setdefault("rings", DEFAULT_RINGS)
        cfg.setdefault("boardDiameterMm", 451.0)

        # Optional nützlich: aktuelle Capture-Settings mitschicken
        res = self.cmb_res.currentData()  # (w, h)
        fps = int(self.cmb_fps.currentData())
        cfg["capture"] = {"resolution": [int(res[0]), int(res[1])], "fps": fps}

        # 3) Neue Werte setzen
        cfg["homographies"] = Hs
        cfg["intrinsics"] = intr_list  # pro Panel k1/k2 + enabled

        # 4) Speichern beim Agent
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


def open_triple_calibration(parent=None):
    dlg = TripleCalibration(parent)
    dlg.exec()
