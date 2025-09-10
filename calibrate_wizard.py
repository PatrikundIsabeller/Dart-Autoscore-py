# calibrate_wizard.py – 3-Kamera-Config + Wizard (Punkte + Feintuning + Save)
from __future__ import annotations
import os, json, time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import requests
from PyQt6 import QtCore, QtGui, QtWidgets

AGENT_URL = os.environ.get("TRIPLEONE_AGENT", "http://127.0.0.1:4700")

DEFAULT_RINGS = {
    "double_outer": 0.995,
    "double_inner": 0.940,
    "triple_outer": 0.620,
    "triple_inner": 0.540,
    "bull_outer": 0.060,
    "bull_inner": 0.015,
}


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


# -------------- Math/Proj -------------
def find_homography(
    board_diam_mm: float, pts_px: List[Tuple[float, float]]
) -> np.ndarray:
    if len(pts_px) != 4:
        raise ValueError("need 4 points (top,right,bottom,left)")
    R = board_diam_mm / 2.0
    world = np.array([[0, -R], [R, 0], [0, R], [-R, 0]], dtype=np.float32)
    img = np.array(pts_px, dtype=np.float32)
    H, _ = cv2.findHomography(world, img, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    return H


def world_circle_points_mm(radius_mm: float, n: int = 360) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius_mm * np.cos(a), radius_mm * np.sin(a)], axis=1).astype(
        np.float32
    )


def project(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    ones = np.ones((pts_xy.shape[0], 1), dtype=np.float32)
    P = np.hstack([pts_xy, ones])
    p = (H @ P.T).T
    return p[:, :2] / p[:, 2:3]


def cvimg_to_qpix(img: np.ndarray) -> QtGui.QPixmap:
    if img is None or img.size == 0:
        return QtGui.QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QPixmap.fromImage(
        QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
    )


# -------------- UI-Helpers ------------
class LiveView(QtWidgets.QWidget):
    """Einfacher OpenCV-Preview pro Slot"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.label = QtWidgets.QLabel(" ")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(360, 220)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)

    def start(self, idx: int, res=(1280, 720), fps=30):
        self.stop()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {idx} not available")
        self.timer.start(33)

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _tick(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        pm = cvimg_to_qpix(frame)
        self.label.setPixmap(
            pm.scaled(
                self.label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def grab(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None
        last = None
        for _ in range(3):
            ok, f = self.cap.read()
            if ok:
                last = f
        return last

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.stop()
        super().closeEvent(e)


class CameraSlot(QtWidgets.QFrame):
    """Ein Slot wie im Screenshot: Preview + 'Select camera' + ↔"""

    activated = QtCore.pyqtSignal(int)  # slot index

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.view = LiveView()
        self.cmb = QtWidgets.QComboBox()
        self.cmb.setPlaceholderText("Select camera")
        for i in range(8):
            self.cmb.addItem(f"Cam {i}", i)

        self.btn_left = QtWidgets.QToolButton()
        self.btn_left.setText("←")
        self.btn_right = QtWidgets.QToolButton()
        self.btn_right.setText("→")

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.cmb, 1)
        bar.addWidget(self.btn_left)
        bar.addWidget(self.btn_right)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)

        # Interaktion
        self.cmb.currentIndexChanged.connect(self._on_select)
        self.mousePressEvent = lambda e: self.activated.emit(self.idx)

        self.current_cam: Optional[int] = None
        self.res = (1280, 720)
        self.fps = 30

    def set_globals(self, res: tuple[int, int], fps: int):
        self.res = res
        self.fps = fps
        if self.current_cam is not None:
            # Restart mit neuen Settings
            try:
                self.view.start(self.current_cam, self.res, self.fps)
            except Exception:
                pass

    def _on_select(self, _):
        cam = self.cmb.currentData()
        if cam is None:
            return
        self.current_cam = int(cam)
        try:
            self.view.start(self.current_cam, self.res, self.fps)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Camera", str(e))

    def grab_frame(self) -> Optional[np.ndarray]:
        return self.view.grab()


# ---------- Feintuning/Wizard ----------
class ClickLabel(QtWidgets.QLabel):
    pointAdded = QtCore.pyqtSignal(float, float)

    def __init__(self, pm: QtGui.QPixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pm)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.points: List[Tuple[int, int]] = []

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self.pixmap():
            return
        pm = self.pixmap()
        s = self.size()
        ps = pm.size()
        scale = min(s.width() / ps.width(), s.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (s.width() - dw) / 2, (s.height() - dh) / 2
        x = (e.position().x() - ox) / scale
        y = (e.position().y() - oy) / scale
        if 0 <= x < ps.width() and 0 <= y < ps.height():
            self.points.append((int(x), int(y)))
            self.pointAdded.emit(x, y)
            self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if not self.pixmap():
            return
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        pm = self.pixmap()
        s = self.size()
        ps = pm.size()
        scale = min(s.width() / ps.width(), s.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (s.width() - dw) / 2, (s.height() - dh) / 2
        for i, (px, py) in enumerate(self.points):
            x = ox + px * scale
            y = oy + py * scale
            p.setPen(QtGui.QPen(QtGui.QColor("#22c55e"), 3))
            p.setBrush(QtGui.QColor(34, 197, 94, 180))
            p.drawEllipse(QtCore.QPointF(x, y), 6, 6)
            p.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb")))
            p.drawText(x + 8, y - 8, f"{i+1}")


class OverlayView(QtWidgets.QWidget):
    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.pm = cvimg_to_qpix(frame)
        self.frame = frame
        self.H: Optional[np.ndarray] = None
        self.diam = 451.0
        self.rings = dict(DEFAULT_RINGS)
        self.setMinimumSize(540, 400)

    def set_params(self, H: np.ndarray, diam: float, rings: dict):
        self.H = H
        self.diam = diam
        self.rings = dict(rings)
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        s = self.size()
        ps = self.pm.size()
        scale = min(s.width() / ps.width(), s.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (s.width() - dw) / 2, (s.height() - dh) / 2
        p.drawPixmap(
            QtCore.QRectF(ox, oy, dw, dh),
            self.pm,
            QtCore.QRectF(0, 0, ps.width(), ps.height()),
        )
        if self.H is None:
            return
        R = self.diam / 2.0
        ring_list = [
            ("double_outer", "#22d3ee"),
            ("double_inner", "#22d3ee"),
            ("triple_outer", "#f59e0b"),
            ("triple_inner", "#f59e0b"),
            ("bull_outer", "#10b981"),
            ("bull_inner", "#ef4444"),
        ]
        for key, col in ring_list:
            r_mm = R * float(self.rings.get(key, DEFAULT_RINGS[key]))
            pts = world_circle_points_mm(r_mm, 360)
            ip = project(self.H, pts)
            path = QtGui.QPainterPath()
            path.moveTo(ip[0, 0] * scale + ox, ip[0, 1] * scale + oy)
            for i in range(1, ip.shape[0]):
                path.lineTo(ip[i, 0] * scale + ox, ip[i, 1] * scale + oy)
            path.closeSubpath()
            pen = QtGui.QPen(QtGui.QColor(col))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawPath(path)


class FineTuneDialog(QtWidgets.QDialog):
    """Frame -> 4 Punkte -> Overlay -> Save"""

    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration – fine tune")
        self.resize(1080, 680)
        self.frame = frame
        self.points: List[Tuple[int, int]] = []
        self.H: Optional[np.ndarray] = None

        # left: click image
        self.click = ClickLabel(cvimg_to_qpix(frame))
        self.click.setMinimumSize(640, 360)

        # right: controls + overlay
        self.overlay = OverlayView(frame)
        self.ed_diam = QtWidgets.QDoubleSpinBox()
        self.ed_diam.setRange(200.0, 600.0)
        self.ed_diam.setDecimals(2)
        self.ed_diam.setValue(451.0)
        self.ed_diam.setSuffix(" mm")
        self.sp = {}
        grid = QtWidgets.QGridLayout()
        labels = [
            ("double_outer", "Double outer"),
            ("double_inner", "Double inner"),
            ("triple_outer", "Triple outer"),
            ("triple_inner", "Triple inner"),
            ("bull_outer", "Bull outer"),
            ("bull_inner", "Bull inner"),
        ]
        for r, (k, t) in enumerate(labels):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(0, 1.2)
            s.setDecimals(4)
            s.setSingleStep(0.001)
            s.setValue(DEFAULT_RINGS[k])
            self.sp[k] = s
            grid.addWidget(QtWidgets.QLabel(t), r, 0)
            grid.addWidget(s, r, 1)

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_cancel = QtWidgets.QPushButton("Discard")
        self.btn_save.setEnabled(False)

        right = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        form.addRow("Board diameter", self.ed_diam)
        right.addLayout(form)
        right.addWidget(self.overlay, 1)
        right.addLayout(grid)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.btn_cancel)
        hb.addStretch(1)
        hb.addWidget(self.btn_save)
        right.addLayout(hb)

        root = QtWidgets.QGridLayout(self)
        root.addWidget(self.click, 0, 0, 1, 1)
        root.addLayout(right, 0, 1, 1, 1)

        # events
        self.click.pointAdded.connect(self._on_point)
        self.ed_diam.valueChanged.connect(self._update_overlay)
        for s in self.sp.values():
            s.valueChanged.connect(self._update_overlay)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_cancel.clicked.connect(self.reject)

        # preload current calib
        cfg = fetch_calibration()
        self.ed_diam.setValue(float(cfg.get("boardDiameterMm", 451.0)))
        rings = cfg.get("rings") or DEFAULT_RINGS
        for k, v in rings.items():
            self.sp[k].setValue(float(v))

    def _on_point(self, *_):
        if len(self.click.points) == 4:
            # Reihenfolge: oben, rechts, unten, links
            try:
                self.H = find_homography(self.ed_diam.value(), self.click.points)
                self.btn_save.setEnabled(True)
                self._update_overlay()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Homography", str(e))

    def _update_overlay(self):
        if self.H is None:
            return
        rings = {k: self.sp[k].value() for k in self.sp}
        self.overlay.set_params(self.H, self.ed_diam.value(), rings)

    def _on_save(self):
        rings = {k: self.sp[k].value() for k in self.sp}
        cfg = fetch_calibration()
        cfg["boardDiameterMm"] = float(self.ed_diam.value())
        cfg["rings"] = rings
        if self.H is not None:
            cfg["homographies"] = [self.H.tolist()]  # TODO: multi-cam erweitern
        try:
            save_calibration(cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save", f"Fehler beim Speichern:\n{e}")
            return
        QtWidgets.QMessageBox.information(self, "Save", "Kalibrierung gespeichert.")
        self.accept()


# -------------- Main Calib Window --------------
class CalibrationWindow(QtWidgets.QDialog):
    """3 Slots + globale Settings + Calibrate"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.resize(1200, 740)

        title = QtWidgets.QLabel("Configure cameras")
        title.setStyleSheet("font-size:24px;font-weight:700;margin:8px 0 16px 0;")

        self.slot = [CameraSlot(0), CameraSlot(1), CameraSlot(2)]
        for s in self.slot:
            s.activated.connect(self._on_active)
        self.active_idx = 0
        self._on_active(0)

        slots = QtWidgets.QHBoxLayout()
        for s in self.slot:
            slots.addWidget(s, 1)

        # Bottom row controls (global)
        self.cmb_res = QtWidgets.QComboBox()
        for w, h in [(1920, 1080), (1600, 900), (1280, 720), (640, 480)]:
            self.cmb_res.addItem(f"{w}x{h}", (w, h))
        self.cmb_res.setCurrentIndex(2)  # 1280x720

        self.cmb_fps = QtWidgets.QComboBox()
        for f in [30, 25, 20, 15]:
            self.cmb_fps.addItem(str(f), f)

        self.cmb_standby = QtWidgets.QComboBox()
        for m in [5, 10, 15, 30, 60]:
            self.cmb_standby.addItem(f"{m} minutes", m)

        self.chk_dist = QtWidgets.QCheckBox("Approximate distortion")
        self.chk_calib_on_change = QtWidgets.QCheckBox("Calibrate on camera change")
        self.chk_calib_on_start = QtWidgets.QCheckBox("Calibrate on startup")

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate")

        row1 = QtWidgets.QGridLayout()
        row1.addWidget(QtWidgets.QLabel("Camera resolution"), 0, 0)
        row1.addWidget(self.cmb_res, 1, 0)
        row1.addWidget(QtWidgets.QLabel("Frames per second"), 0, 1)
        row1.addWidget(self.cmb_fps, 1, 1)
        row1.addWidget(QtWidgets.QLabel("Camera standby time"), 0, 2)
        row1.addWidget(self.cmb_standby, 1, 2)
        row1.addItem(
            QtWidgets.QSpacerItem(
                20,
                20,
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Minimum,
            ),
            1,
            3,
        )
        row1.addWidget(self.chk_dist, 1, 4)
        row1.addWidget(self.chk_calib_on_change, 1, 5)
        row1.addWidget(self.chk_calib_on_start, 1, 6)
        row1.addWidget(self.btn_calibrate, 1, 7)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(title)
        root.addLayout(slots)
        root.addSpacing(6)
        root.addLayout(row1)

        # wiring
        self.cmb_res.currentIndexChanged.connect(self._apply_globals)
        self.cmb_fps.currentIndexChanged.connect(self._apply_globals)
        self.btn_calibrate.clicked.connect(self._run_calibration)

        self._apply_globals()

    def _on_active(self, idx: int):
        self.active_idx = idx
        # markiere aktiv visuell
        for i, s in enumerate(self.slot):
            s.setStyleSheet(
                "border:2px solid #3b82f6;border-radius:12px;"
                if i == idx
                else "border:1px solid #e5e7eb;border-radius:12px;"
            )

    def _apply_globals(self):
        res = self.cmb_res.currentData()
        fps = int(self.cmb_fps.currentData())
        for s in self.slot:
            s.set_globals(res, fps)

    def _run_calibration(self):
        s = self.slot[self.active_idx]
        frame = s.grab_frame()
        if frame is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Kein Kamerabild. Bitte Kamera wählen / Vorschau läuft?",
            )
            return
        dlg = FineTuneDialog(frame, self)
        dlg.exec()


# ------- Convenience entry for controller -------
def open_calibration(parent=None):
    dlg = CalibrationWindow(parent)
    dlg.exec()
