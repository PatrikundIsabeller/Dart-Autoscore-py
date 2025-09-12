# triple_calib.py – 3-Kamera-Kalibrierung (PyQt6) – autodarts 4-Punkt, Welt→Bild Homographie
from __future__ import annotations
import os, json
from typing import List, Tuple, Optional

import cv2
import numpy as np
import requests
from PyQt6 import QtCore, QtGui, QtWidgets

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
    gray: np.ndarray, pts_img: np.ndarray, search_px: int = 6
) -> np.ndarray:
    """
    Subpixel-Refinement: verschiebt jeden Punkt radial (vom Mittelpunkt aus) auf das Maximum
    der Bildintensitäts-Gradienten (Double-Outer-Kante).
    """
    assert gray.ndim == 2
    assert pts_img.shape == (4, 2)
    # Mittelpunkt grob als Mittel der 4 Punkte
    c = np.mean(pts_img, axis=0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]

    out = []
    for x, y in pts_img:
        v = np.array([x, y], dtype=np.float32) - c
        n = float(np.linalg.norm(v) + 1e-6)
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


# ===========
# 2) HTTP I/O
# ===========


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


def make_grid600_image(include_numbers=True) -> np.ndarray:
    """
    Erzeugt 600×600 Overlay (BGR):
      - Ringe (autodarts-Radien)
      - 20 Sektor-Grenzen (k*18°)
      - Zahlen in **Segmentmitte** (+9°), leicht außerhalb Double-Outer
    """
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    cx = cy = 300
    R = 300.0

    teal = (0, 255, 255)

    def circ(rel, color, thickness=2):
        cv2.circle(img, (cx, cy), int(rel * R), color, thickness, cv2.LINE_AA)

    # Ringe
    circ(REL["double_outer"], teal, 2)
    circ(REL["double_inner"], teal, 1)
    circ(REL["triple_outer"], teal, 2)
    circ(REL["triple_inner"], teal, 1)
    circ(REL["bull_outer"], (255, 255, 0), 1)
    circ(REL["bull_inner"], (255, 0, 0), -1)

    # Sektorgrenzen
    for k in range(20):
        ang_deg = -90 + k * 18
        t = np.deg2rad(ang_deg)
        r0 = REL["bull_outer"] * R
        r1 = REL["double_outer"] * R
        x0, y0 = int(cx + r0 * np.cos(t)), int(cy + r0 * np.sin(t))
        x1, y1 = int(cx + r1 * np.cos(t)), int(cy + r1 * np.sin(t))
        cv2.line(img, (x0, y0), (x1, y1), teal, 1, cv2.LINE_AA)

    # Zahlen mittig im Segment (+9°)
    if include_numbers:
        rel_number = REL["double_outer"] * 1.06  # leicht außerhalb
        r_txt = rel_number * R
        font = cv2.FONT_HERSHEY_SIMPLEX
        for k, num in enumerate(SECTOR_ORDER):
            ang_deg = -90 + (k * 18 + 9)
            t = np.deg2rad(ang_deg)
            x = cx + r_txt * np.cos(t)
            y = cy + r_txt * np.sin(t)
            text = str(num)
            scale = 0.55
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            org = (int(x - tw / 2), int(y + th / 2))
            cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(
                img, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA
            )
    return img


GRID_600 = make_grid600_image(True)


# ==================
# 4) Camera Utilities
# ==================


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


def cvimg_to_qpix(img: np.ndarray) -> QtGui.QPixmap:
    if img is None or img.size == 0:
        return QtGui.QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QPixmap.fromImage(
        QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
    )


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

        # State
        self.frame: Optional[np.ndarray] = None
        # Startpunkte: grob um das Bild platziert
        self.points = np.array(
            [[300, 120], [540, 300], [300, 480], [60, 300]], dtype=np.float32
        )
        self._drag_idx = -1
        self._active_idx = 0

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
            self.pointsChanged.emit()
            self.update()

    # ----- Homography compute (Weltmodus) -----
    def homography(self) -> Optional[np.ndarray]:
        """
        Rechnet (bei vorhandenem Frame) die Welt- und Template-Homographien aus self.points.
        """
        if self.frame is None or self.points.shape != (4, 2):
            return None
        try:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            H_world, H_tpl, pts_ref = compute_homographies_from_points(
                gray, self.points
            )
            self.points = pts_ref
            self.H_world2img = H_world
            self.H_tplPx2img = H_tpl
            return self.H_tplPx2img
        except Exception:
            return None

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

        # Overlay: warped Grid via H_tplPx2img
        H_tpl = self.homography()
        if H_tpl is not None:
            h, w = self.frame.shape[:2]
            warped = cv2.warpPerspective(
                GRID_600,
                H_tpl,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            wpm = cvimg_to_qpix(warped)
            p.setOpacity(self.alpha)
            self._paint_pixmap(p, wpm)
            p.setOpacity(1.0)

            # rotes Wedge via Homographie
            if self._wedge_tmpl is not None:
                try:
                    pts_img = cv2.perspectiveTransform(self._wedge_tmpl, H_tpl).reshape(
                        -1, 2
                    )
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

        # 4 grüne Punkte
        p.setPen(QtGui.QPen(QtGui.QColor("#22c55e"), 3))
        p.setBrush(QtGui.QColor(34, 197, 94, 180))
        fw, fh = self.frame.shape[1], self.frame.shape[0]
        labels = ["20-1", "6-10", "3-19", "11-14"]
        for idx, (x_img, y_img) in enumerate(self.points):
            x = ox + (x_img / fw) * (ps.width() * scale)
            y = oy + (y_img / fh) * (ps.height() * scale)
            p.drawEllipse(QtCore.QPointF(float(x), float(y)), 6, 6)
            p.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
            p.drawText(QtCore.QPointF(float(x) + 8.0, float(y) - 8.0), labels[idx])
            p.setPen(QtGui.QPen(QtGui.QColor("#22c55e"), 3))

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
        if self.frame is None:
            return
        xi, yi = self._img_coords_from_mouse(e)
        d = np.linalg.norm(self.points - np.array([xi, yi]), axis=1)
        self._drag_idx = int(np.argmin(d)) if d.size else -1
        if self._drag_idx >= 0:
            self._active_idx = self._drag_idx

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.frame is None or self._drag_idx < 0:
            return
        xi, yi = self._img_coords_from_mouse(e)
        self.points[self._drag_idx] = [xi, yi]
        self.pointsChanged.emit()
        self.update()

    def mouseReleaseEvent(self, _e: QtGui.QMouseEvent):
        self._drag_idx = -1

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
        self.pointsChanged.emit()
        self.update()


# ===================
# 6) CamPanel (1 Cam)
# ===================


class CamPanel(QtWidgets.QFrame):
    """
    Eine Kamera-Kachel:
      - Kamera wählen
      - Undistort (k1/k2)
      - Auto (grober Seed)
      - 4-Punkt-Klick (Double-Outer) → Welt→Bild Homographie
    """

    def __init__(self, cam_id_guess: int):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # Ansicht
        self.view = DragBoard()
        self.view.set_wedge_template(90.0)  # 20 nach oben im Template

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

        # (Optional) Nudge-Buttons ±18° fürs Wedge (rein visuell)
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
        self.cb_undist.stateChanged.connect(lambda *_: self.view.update())
        self.k1.valueChanged.connect(lambda *_: self.view.update())
        self.k2.valueChanged.connect(lambda *_: self.view.update())
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
                self, "Camera", f"Cannot open camera {cam_id} on any backend."
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
        # sorgt dafür, dass H intern berechnet ist
        return self.view.homography()

    def current_world_h(self) -> Optional[np.ndarray]:
        self.view.homography()
        return self.view.H_world2img

    def current_points(self) -> list[list[float]]:
        return self.view.points.tolist()

    # ---------- Auto (grober Seed für 4 Punkte) ----------
    def auto_calibrate(self):
        """
        Grobe Auto-Punktvorschläge: nutzt einfache Kreis/ellipse-Annäherung,
        setzt die 4 Punkte (oben/rechts/unten/links) auf den Double-Outer.
        Danach verfeinert die DragBoard-Homography() ohnehin noch subpixel.
        """
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

        work = frame.copy()
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        edges = cv2.Canny(gray, 60, 140)

        # Ellipse fitten aus Kanten (Fallback auf größten Kreis)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best = None
        best_score = -1
        H, W = gray.shape[:2]
        area_img = H * W
        for c in cnts:
            if len(c) < 50:
                continue
            area = cv2.contourArea(c)
            if area < 0.02 * area_img:
                continue
            try:
                (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
            except cv2.error:
                continue
            ar = min(MA, ma) / max(MA, ma)
            score = area * (0.5 + 0.5 * ar)
            if score > best_score:
                best_score, best = score, ((cx, cy), (MA, ma), ang)
        if best is None:
            QtWidgets.QMessageBox.warning(self, "Auto", "Board-Ellipse nicht gefunden.")
            return

        (cx, cy), (MA, ma), ang = best
        a, b = MA / 2.0, ma / 2.0

        def ellipse_radius_at(img_angle_rad: float, theta_deg: float) -> float:
            theta = np.deg2rad(theta_deg)
            phi_p = img_angle_rad - theta
            c, s = np.cos(phi_p), np.sin(phi_p)
            den = (c * c) / (a * a) + (s * s) / (b * b)
            return 0.0 if den <= 1e-12 else 1.0 / np.sqrt(den)

        # 4 Zielwinkel in Bildkoordinaten (oben/rechts/unten/links) an *Grenzlinien* (+9°)
        desired_deg = [
            -81.0,
            9.0,
            99.0,
            -171.0,
        ]  # (-90 + 9), (0+9), (90+9), (180+9-360)
        pts = []
        for deg in desired_deg:
            phi = np.deg2rad(deg)
            r = ellipse_radius_at(phi, ang) * REL["double_outer"]  # Double-Outer Anteil
            x = cx + r * np.cos(phi)
            y = cy + r * np.sin(phi)
            pts.append([x, y])

        self.view.set_points(np.array(pts, dtype=np.float32))

    def _nudge_wedge(self, delta_deg: float):
        """Dreht *nur* das Wedge-Template (±18° / Feinnudge)."""
        if hasattr(self.view, "_wedge_angle_deg"):
            self.view.set_wedge_template(self.view._wedge_angle_deg + delta_deg)


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
        self._apply_globals()

    def _apply_globals(self):
        res = self.cmb_res.currentData()
        fps = int(self.cmb_fps.currentData())
        for p in self.panels:
            p.set_globals(res, fps)

    def _tick(self):
        for p in self.panels:
            p.tick()

    # --- Save/Load (JSON) ---
    def _save_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration", "calibration.json", "JSON (*.json)"
        )
        if not path:
            return
        data = {}
        for i, p in enumerate(self.panels, start=1):
            H_tpl = p.current_overlay_h()
            H_w = p.current_world_h()
            if H_tpl is None or H_w is None:
                continue
            data[f"camera_{i}"] = {
                "points_img": p.current_points(),
                "H_overlay_tplPx2img": H_tpl.tolist(),
                "H_world2img": H_w.tolist(),
                "intrinsics": {"k1": float(p.k1.value()), "k2": float(p.k2.value())},
                "board_diameter_mm": 451.0,
                "rel_radii": REL,
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
                    pts = np.array(data[key].get("points_img"), dtype=np.float32)
                    if pts.shape == (4, 2):
                        p.view.set_points(pts)
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
        Hs_overlay = []
        Hs_world = []
        intr_list = []
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


def open_triple_calibration(parent=None):
    dlg = TripleCalibration(parent)
    dlg.exec()
