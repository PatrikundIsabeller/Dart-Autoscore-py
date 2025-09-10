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


def draw_dartboard_grid(size: int = 600) -> np.ndarray:
    """Einfaches Raster (Double/Triple/Bull + Segmentlinien + Zahlen)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    R = size // 2
    white = (255, 255, 255)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    # double
    cv2.circle(img, (cx, cy), int(R * 0.95), red, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * 1.00), red, 2, cv2.LINE_AA)
    # triple
    cv2.circle(img, (cx, cy), int(R * 0.60), green, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * 0.66), green, 2, cv2.LINE_AA)
    # bull
    cv2.circle(img, (cx, cy), int(R * 0.06), blue, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(R * 0.12), yellow, 2, cv2.LINE_AA)
    # Segmente + Zahlen
    nums = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    for i in range(20):
        ang = np.deg2rad(90 - i * 18)
        x0 = int(cx + R * 0.12 * np.cos(ang))
        y0 = int(cy - R * 0.12 * np.sin(ang))
        x1 = int(cx + R * 1.00 * np.cos(ang))
        y1 = int(cy - R * 1.00 * np.sin(ang))
        cv2.line(img, (x0, y0), (x1, y1), white, 1, cv2.LINE_AA)
        txt = str(nums[i])
        rad = R * 0.85
        tx = int(cx + rad * np.cos(ang))
        ty = int(cy - rad * np.sin(ang))
        sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(
            img,
            (tx - sz[0] // 2 - 5, ty - sz[1] // 2 - 5),
            (tx + sz[0] // 2 + 5, ty + sz[1] // 2 + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            txt,
            (tx - sz[0] // 2, ty + sz[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            white,
            2,
            cv2.LINE_AA,
        )
    return img


GRID_600 = draw_dartboard_grid(600)


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
    """Zeigt Liveframe + warped Grid + 4 draggable Punkte (TL,TR,BR,BL)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setStyleSheet(
            "QLabel{border:1px solid #404040;border-radius:8px;background:#111;color:#fff;}"
        )
        # state
        self.frame: Optional[np.ndarray] = None
        self.points = np.array(
            [[100, 100], [540, 100], [540, 380], [100, 380]], dtype=np.float32
        )
        self.M: Optional[np.ndarray] = None

    def set_frame(self, frame: Optional[np.ndarray]):
        self.frame = frame
        self.update()

    def homography(self) -> Optional[np.ndarray]:
        if self.points.shape != (4, 2):
            return None
        # Board-Gitter (600x600) -> Bildpunkte
        src = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
        dst = self.points.astype(np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst)
        return self.M

    def _paint_pixmap(self, p: QtGui.QPainter, pm: QtGui.QPixmap):
        rect = self.rect()
        ps = pm.size()
        scale = min(rect.width() / ps.width(), rect.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (rect.width() - dw) / 2, (rect.height() - dh) / 2
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

        # Overlay: warped Grid
        if self.homography() is not None:
            h, w = self.frame.shape[:2]
            warped = cv2.warpPerspective(GRID_600, self.M, (w, h))
            wpm = cvimg_to_qpix(warped)
            p.setOpacity(0.7)
            self._paint_pixmap(p, wpm)
            p.setOpacity(1.0)

        # Punkte
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

    def _img_coords_from_mouse(self, e: QtGui.QMouseEvent):
        pm = cvimg_to_qpix(self.frame)
        rect = self.rect()
        ps = pm.size()
        scale = min(rect.width() / ps.width(), rect.height() / ps.height())
        dw, dh = ps.width() * scale, ps.height() * scale
        ox, oy = (rect.width() - dw) / 2, (rect.height() - dh) / 2
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

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.frame is None or getattr(self, "_drag_idx", -1) < 0:
            return
        xi, yi = self._img_coords_from_mouse(e)
        self.points[self._drag_idx] = [xi, yi]
        self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self._drag_idx = -1


# -------------- Kamera-Panel (eine Kachel) --------------
class CamPanel(QFrame):
    """Eine Kalibrier-Kachel: Kameraauswahl, optionales Entzerren, Live-View mit DragBoard."""

    def __init__(self, cam_id_guess: int):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)

        # Bildfläche
        self.view = DragBoard()

        # --- Leiste unten -----------------------------------------------------
        bar = QtWidgets.QHBoxLayout()

        # Kamera-Dropdown (kein Autostart)
        self.cmb = QtWidgets.QComboBox()
        self.cmb.addItem("Select camera", None)
        bar.addWidget(self.cmb, 1)

        # Entzerrung (approx. radial: k1/k2)
        self.cb_undist = QtWidgets.QCheckBox("Approx. distortion")
        self.k1 = QtWidgets.QDoubleSpinBox()
        self.k1.setRange(-0.50, 0.50)
        self.k1.setDecimals(3)
        self.k1.setSingleStep(0.01)
        self.k1.setValue(0.00)
        self.k1.setPrefix("k1 ")
        self.k2 = QtWidgets.QDoubleSpinBox()
        self.k2.setRange(-0.50, 0.50)
        self.k2.setDecimals(3)
        self.k2.setSingleStep(0.01)
        self.k2.setValue(0.00)
        self.k2.setPrefix("k2 ")
        bar.addWidget(self.cb_undist, 0)
        bar.addWidget(self.k1, 0)
        bar.addWidget(self.k2, 0)

        # Layout zusammensetzen
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)

        # Status
        self.cap: Optional[cv2.VideoCapture] = None
        self.res = (1280, 720)
        self.fps = 30

        # Signals
        self.cmb.currentIndexChanged.connect(self._on_select)
        self.cb_undist.stateChanged.connect(lambda *_: self.view.update())
        self.k1.valueChanged.connect(lambda *_: self.view.update())
        self.k2.valueChanged.connect(lambda *_: self.view.update())

    # --- Kamera Handling ------------------------------------------------------
    def _on_select(self, _):
        cam = self.cmb.currentData()
        if cam is None:
            self.stop()
            return
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
        self.view.set_frame(frame.copy())

    def set_globals(self, res: tuple[int, int], fps: int):
        self.res = res
        self.fps = fps
        if self.cap:
            idx = self.cmb.currentData()
            if idx is not None:
                self.start(int(idx))

    # --- Zugriff für Save/Export ---------------------------------------------
    def current_h(self) -> Optional[np.ndarray]:
        return self.view.homography()

    def current_points(self) -> List[List[float]]:
        return self.view.points.tolist()


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
