# controller_app.py
# -------------------------------------------------------------
# Python/PyQt6 Desktop-"Backend" wie dein Screenshot (onlinefähig)
# Jetzt mit: Agent-URL, Status-Poller (HTTP), Buttons -> HTTP-Calls
# - Links: Controls/Detection/Updates/Preferences/Links
# - Rechts: Dartboard-Preview
# - Unten: Statusbar (FPS/CPU/MEM/Version) wird 1×/s aktualisiert
#
# Start:
#   1) venv aktivieren
#   2) pip install PyQt6 requests
#   3) python controller_app.py
#
# Agent (Mock) separat starten (siehe Chat-Anleitung): http://127.0.0.1:4700
# -------------------------------------------------------------
from triple_calib import open_triple_calibration
from dotenv import load_dotenv

load_dotenv()  # lädt .env
from PyQt6 import QtCore, QtGui, QtWidgets
from calibration_dialog import CalibrationDialog
from calibrate_wizard import open_calibration

# controller_app.py
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QPushButton
from play_launcher import open_play_dashboard, is_calibration_valid  # ← reuse
import webbrowser
import random
import requests
import time
import os
import math


APP_VERSION = "v0.1.0"
DETECTOR_VERSION = "v0.26.15"
AGENT_URL = os.environ.get("TRIPLEONE_AGENT", "http://127.0.0.1:4700")
API_URL = os.environ.get("TRIPLEONE_API", "http://127.0.0.1:5080")
DEVICE_ID = os.environ.get("TRIPLEONE_DEVICE", "dev_local")


# ----------------------------- Status-Poller ----------------------------- #
class StatusPoller(QtCore.QThread):
    """Fragt periodisch /status beim lokalen Agent ab und sendet ein Signal."""

    status = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, base_url: str, interval_sec: float = 1.0, parent=None):
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.interval = interval_sec
        self._running = True

    def run(self):
        sess = requests.Session()
        while self._running:
            try:
                r = sess.get(self.base_url + "/status", timeout=0.7)
                if r.ok:
                    self.status.emit(r.json())
                else:
                    self.error.emit(f"HTTP {r.status_code}")
            except Exception as e:
                self.error.emit(str(e))
            time.sleep(self.interval)

    def stop(self):
        self._running = False


# ------------------------------- UI-Teile ------------------------------- #
class Card(QtWidgets.QFrame):
    def __init__(self, title: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        if title is not None:
            head = QtWidgets.QLabel(title)
            head.setObjectName("CardTitle")
            lay.addWidget(head)
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            sep.setObjectName("CardSep")
            lay.addWidget(sep)
        self.body = QtWidgets.QWidget()
        self.body_lay = QtWidgets.QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        self.body_lay.setSpacing(10)
        lay.addWidget(self.body)


class StatePill(QtWidgets.QPushButton):
    STATES = ["IDLE", "ARMED", "THROW", "SETTLING"]

    def __init__(self, parent=None):
        super().__init__("THROW", parent)
        self.setObjectName("StatePill")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.state_index = 2  # "THROW"
        self.clicked.connect(self._cycle)
        self._apply_style()

    def set_state(self, s: str):
        if s in self.STATES:
            self.state_index = self.STATES.index(s)
            self.setText(s)
            self._apply_style()

    def _cycle(self):
        self.state_index = (self.state_index + 1) % len(self.STATES)
        self.setText(self.STATES[self.state_index])
        self._apply_style()

    def _apply_style(self):
        s = self.STATES[self.state_index]
        color = {
            "THROW": "#34d399",  # emerald-400
            "SETTLING": "#f59e0b",  # amber-500
            "ARMED": "#818cf8",  # indigo-400
            "IDLE": "#cbd5e1",  # slate-300
        }[s]
        # Direkter Stylesheet (QSS attr() gibt es nicht zuverlässig)
        self.setStyleSheet(
            f"QPushButton#StatePill{{background:{color};color:#0b1220;border-radius:10px;padding:8px 14px;font-weight:600;}}"
        )


class DartboardPreview(QtWidgets.QWidget):
    def sizeHint(self):
        return QtCore.QSize(520, 520)

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        rect = self.rect()
        s = min(rect.width(), rect.height())
        cx = rect.x() + rect.width() // 2
        cy = rect.y() + rect.height() // 2
        R = int(s * 0.46)
        center = QtCore.QPoint(cx, cy)

        # Hintergrund
        p.fillRect(rect, QtGui.QColor("#0b1220"))

        # Boardgrund
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#000000"))
        p.drawEllipse(center, R, R)

        # Segmente (20 Strahlen) -> math statt QtCore.qCos/qSin
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 2)
        p.setPen(pen)
        for i in range(20):
            angle_deg = (360 / 20) * i
            rad = math.radians(angle_deg)
            x = cx + int(R * 0.92 * math.cos(rad))
            y = cy + int(R * 0.92 * math.sin(rad))
            p.drawLine(center, QtCore.QPoint(x, y))

        # Ring-Helper
        def ring(outer: float, inner: float, color: str):
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(color))
            p.drawEllipse(center, int(R * outer), int(R * outer))
            p.setBrush(QtGui.QColor("#0b1220"))
            p.drawEllipse(center, int(R * inner), int(R * inner))

        # Double / Triple
        ring(0.96, 0.88, "#10b981b3")
        ring(0.60, 0.52, "#f43f5eb3")

        # Singles (vereinfacht)
        for i in range(20):
            start_angle = int((16 * 360 / 20) * i)
            span = int(16 * 360 / 20)
            color = QtGui.QColor("#f4f1de") if i % 2 else QtGui.QColor("#111111")
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawPie(
                cx - int(R * 0.62),
                cy - int(R * 0.62),
                int(2 * R * 0.62),
                int(2 * R * 0.62),
                start_angle,
                span,
            )

        # Bulls
        p.setBrush(QtGui.QColor("#10b981"))
        p.drawEllipse(center, int(R * 0.06), int(R * 0.06))
        p.setBrush(QtGui.QColor("#f43f5e"))
        p.setPen(QtGui.QPen(QtGui.QColor("#000"), 2))
        p.drawEllipse(center, int(R * 0.022), int(R * 0.022))


class FooterBar(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Footer")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)

        left = QtWidgets.QHBoxLayout()
        left.setSpacing(6)
        for text in ["Throw", "🖥️", "🔒"]:
            chip = QtWidgets.QLabel(text)
            chip.setProperty("class", "chip")
            left.addWidget(chip)
        left_box = QtWidgets.QWidget()
        left_box.setLayout(left)

        right = QtWidgets.QHBoxLayout()
        right.setSpacing(12)
        self.lbl_fps = QtWidgets.QLabel("– FPS")
        self.lbl_drop = QtWidgets.QLabel("0 Dropped")
        self.lbl_mem = QtWidgets.QLabel("– MB")
        self.lbl_cpu = QtWidgets.QLabel("–%")
        self.lbl_app = QtWidgets.QLabel(APP_VERSION)
        self.lbl_det = QtWidgets.QLabel(DETECTOR_VERSION)
        for w in [
            self.lbl_fps,
            self.lbl_drop,
            self.lbl_mem,
            self.lbl_cpu,
            self.lbl_app,
            self.lbl_det,
        ]:
            w.setProperty("class", "stat")
            right.addWidget(w)
        right.addStretch(1)
        right_box = QtWidgets.QWidget()
        right_box.setLayout(right)

        main = QtWidgets.QHBoxLayout()
        main.addWidget(left_box)
        main.addStretch(1)
        main.addWidget(right_box)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(main)
        lay.addWidget(wrapper)


class TopBar(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(12)

        logo = QtWidgets.QLabel()
        pix = QtGui.QPixmap(22, 22)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QColor("#818cf8"))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 20, 20)
        painter.setBrush(QtGui.QColor(255, 255, 255, 80))
        painter.drawEllipse(5, 5, 12, 12)
        painter.end()
        logo.setPixmap(pix)
        title = QtWidgets.QLabel("AUTODARTS")
        title.setProperty("class", "title")

        left = QtWidgets.QHBoxLayout()
        left.setSpacing(8)
        left.addWidget(logo)
        left.addWidget(title)
        left_box = QtWidgets.QWidget()
        left_box.setLayout(left)

        crumb = QtWidgets.QLabel("🏠 / Desktop / Detection")
        crumb.setProperty("class", "crumb")

        online = QtWidgets.QLabel("Online")
        online.setProperty("class", "kbd")
        user = QtWidgets.QLabel("MAXINATOR1")
        user.setProperty("class", "user")

        right = QtWidgets.QHBoxLayout()
        right.setSpacing(10)
        right.addWidget(online)
        right.addWidget(user)
        right_box = QtWidgets.QWidget()
        right_box.setLayout(right)

        lay.addWidget(left_box)
        lay.addStretch(1)
        lay.addWidget(crumb)
        lay.addStretch(1)
        lay.addWidget(right_box)


class MainWindow(QtWidgets.QMainWindow):
    """
    Hauptfenster des Controllers:
    - Links: Controls/Detection/Updates/Preferences/Links
    - Rechts: Dartboard-Preview
    - Unten: Statusbar
    - Pollt 1×/s den lokalen Agent (/status)
    - on_play: Snapshot -> Session -> Play-URL mit One-Time-Token öffnen
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Triple One – Detection")
        self.resize(1280, 800)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Topbar
        self.topbar = TopBar()
        root.addWidget(self.topbar)

        # Main Grid
        main = QtWidgets.QHBoxLayout()
        main.setContentsMargins(12, 0, 12, 0)
        main.setSpacing(12)

        # ---- Linke Spalte -------------------------------------------------
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(12)

        # Controls
        c_controls = Card("Controls")
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        for label, cb in [
            ("Restart", self.on_restart),
            ("Stop", self.on_stop),
            ("Reset", self.on_reset),
            ("Calibrate", self.on_calibrate),
        ]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(cb)
            b.setProperty("class", "btn")
            row.addWidget(b)
        c_controls.body_lay.addLayout(row)
        left_col.addWidget(c_controls)

        # Detection
        c_detection = Card("Detection")
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)
        self.pill = StatePill()
        row2.addWidget(self.pill)
        for _ in range(3):
            slot = QtWidgets.QLabel("–")
            slot.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            slot.setProperty("class", "slot")
            row2.addWidget(slot)
        c_detection.body_lay.addLayout(row2)
        left_col.addWidget(c_detection)

        # Updates
        c_updates = Card("Detection software updates")
        hl = QtWidgets.QHBoxLayout()
        badge = QtWidgets.QLabel("Up to date")
        badge.setProperty("class", "badge-green")
        self.lbl_update = QtWidgets.QLabel(
            f"{DETECTOR_VERSION} is currently the newest version available."
        )
        hl.addWidget(badge)
        hl.addWidget(self.lbl_update)
        hl.addStretch(1)
        c_updates.body_lay.addLayout(hl)
        btn_check = QtWidgets.QPushButton("Check for updates")
        btn_check.clicked.connect(lambda: self.toast("Check for updates… (Demo)"))
        btn_check.setProperty("class", "btn-outline")
        c_updates.body_lay.addWidget(btn_check)
        left_col.addWidget(c_updates)

        # Preferences
        c_prefs = Card("Preferences")
        cb1 = QtWidgets.QCheckBox(
            "Start Autodarts Desktop when you sign in to your computer"
        )
        cb2 = QtWidgets.QCheckBox("Automatically install detection updates")
        cb2.setChecked(True)
        c_prefs.body_lay.addWidget(cb1)
        c_prefs.body_lay.addWidget(cb2)
        left_col.addWidget(c_prefs)

        # External Links  (Play ruft jetzt self.on_play auf!)
        c_links = Card("External Links")
        links = QtWidgets.QHBoxLayout()

        btn_play = QtWidgets.QPushButton("Play")
        btn_play.setProperty("class", "btn-link")
        btn_play.clicked.connect(self.on_play)  # <<< wichtig
        links.addWidget(btn_play)

        for name, url in [
            ("Docs", "https://docs.example.local"),
            ("Discord", "https://discord.gg/example"),
        ]:
            btn = QtWidgets.QPushButton(name)
            btn.setProperty("class", "btn-link")
            btn.clicked.connect(lambda _, u=url: webbrowser.open(u))
            links.addWidget(btn)

        c_links.body_lay.addLayout(links)
        left_col.addWidget(c_links)

        left_wrap = QtWidgets.QWidget()
        left_wrap.setLayout(left_col)
        left_wrap.setMinimumWidth(360)
        left_wrap.setMaximumWidth(540)

        # ---- Rechte Spalte ------------------------------------------------
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(12)
        preview_card = Card(None)
        self.preview = DartboardPreview()
        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.preview, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addStretch(1)
        preview_card.body_lay.addLayout(vbox)
        right_col.addWidget(preview_card)

        right_wrap = QtWidgets.QWidget()
        right_wrap.setLayout(right_col)

        main.addWidget(left_wrap, 5)
        main.addWidget(right_wrap, 7)
        root.addLayout(main, 1)

        # Footer
        self.footer = FooterBar()
        root.addWidget(self.footer)

        self.setCentralWidget(central)
        self.apply_style()

        # ---- Status-Poller starten ---------------------------------------
        self.poller = StatusPoller(
            os.environ.get(
                "TRIPLEONE_AGENT",
                AGENT_URL if "AGENT_URL" in globals() else "http://127.0.0.1:4700",
            ),
            1.0,
            self,
        )
        self.poller.status.connect(self.on_status)
        self.poller.error.connect(self.on_status_error)
        self.poller.start()

    # ---------------------- HTTP Helper & Actions -------------------------
    def http_post(self, path: str, body: dict | None = None):
        try:
            base = os.environ.get(
                "TRIPLEONE_AGENT",
                AGENT_URL if "AGENT_URL" in globals() else "http://127.0.0.1:4700",
            )
            r = requests.post(base.rstrip("/") + path, json=body or {}, timeout=1.2)
            if r.ok:
                if r.headers.get("content-type", "").startswith("application/json"):
                    return r.json()
                return True
            self.toast(f"HTTP {r.status_code} – {r.text[:200]}")
        except Exception as e:
            self.toast(f"Request fehlgeschlagen: {e}")
        return None

    def on_restart(self):
        self.http_post("/control/restart")

    def on_stop(self):
        self.http_post("/control/stop")

    def on_reset(self):
        self.http_post("/control/reset")

    def on_calibrate(self):
        open_triple_calibration(self)

    # ------------------------- PLAY-FLOW ----------------------------------
    # In class MainWindow:

    def on_play(self):
        """
        Öffnet das lokale Play-Dashboard im Standard-Browser.
        (Nutzt play_launcher.open_play_dashboard)
        """
        if not is_calibration_valid():
            QMessageBox.warning(
                self, "Kalibrierung fehlt", "Bitte zuerst die Kalibrierung abschließen."
            )
            # return  # auskommentieren, wenn du trotzdem öffnen willst

        # Spiele-Grid öffnen:
        open_play_dashboard()

        # Optional: direkt in X01-Setup springen:
        # open_play_dashboard("x01")  # -> öffnet ...#/setup?mode=x01

    # ------------------------- Status/Styling ------------------------------
    def on_status(self, data: dict):
        state = data.get("state", "IDLE")
        self.pill.set_state(state)
        fps = data.get("fps", 0)
        cpu = data.get("cpu", 0.0)
        mem = data.get("mem", 0.0)
        self.footer.lbl_fps.setText(f"{fps} FPS")
        self.footer.lbl_cpu.setText(f"{cpu:.2f}%")
        self.footer.lbl_mem.setText(f"{mem:.2f} MB")
        detv = data.get("versions", {}).get("detector")
        if detv:
            self.footer.lbl_det.setText(detv)

    def on_status_error(self, msg: str):
        # dezente Anzeige, kein Popup
        self.footer.lbl_fps.setText("– FPS")
        self.footer.lbl_cpu.setText("–%")
        self.footer.lbl_mem.setText("– MB")

    def toast(self, msg: str):
        QtWidgets.QMessageBox.information(self, "Info", msg)

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.poller.stop()
            self.poller.wait(800)
        finally:
            super().closeEvent(e)

    def apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0f172a, stop:1 #1e1b4b); }
            #TopBar { border-bottom: 1px solid rgba(255,255,255,0.1); background: rgba(2,6,23,0.6); }
            #TopBar .title { color: #e5e7eb; font-weight: 600; letter-spacing: 1px; }
            #TopBar .crumb { color: #cbd5e1; }
            #TopBar .user { color: #e5e7eb; font-weight: 500; }
            #TopBar .kbd { background: rgba(255,255,255,0.1); color:#e5e7eb; padding:2px 6px; border-radius:6px; font-size:12px; }

            #Card, QWidget[class="card"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; }
            QLabel#CardTitle { color: #cbd5e1; font-weight: 600; }
            #CardSep { color: rgba(255,255,255,0.1); background: rgba(255,255,255,0.1); height:1px; }

            QPushButton[class="btn"] { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px; padding: 8px 14px; color:#e5e7eb; }
            QPushButton[class="btn"]:hover { background: rgba(255,255,255,0.1); }
            QPushButton[class="btn-outline"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15);
                border-radius: 10px; padding: 8px 12px; color:#e5e7eb; }
            QPushButton[class="btn-outline"]:hover { background: rgba(255,255,255,0.1); }
            QPushButton[class="btn-link"] { background: #0f172a; border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px; padding: 8px 14px; color:#e5e7eb; }
            QPushButton[class="btn-link"]:hover { background:#1f2937; }

            QLabel[class="slot"] { min-width: 64px; min-height: 40px; background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; }
            QLabel[class="badge-green"] { background: rgba(16,185,129,0.2); color: #a7f3d0; border:1px solid rgba(16,185,129,0.3);
                border-radius: 8px; padding: 2px 8px; }

            #Footer { border-top: 1px solid rgba(255,255,255,0.1); background: rgba(2,6,23,0.7); }
            #Footer .chip { background: rgba(16,185,129,0.3); color:#d1fae5; padding:2px 6px; border-radius:6px; }
            #Footer .stat { color:#94a3b8; }
            """
        )


# ------------------------------- Main ---------------------------------- #
def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Triple One Controller")
    app.setOrganizationName("Triple One")

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
