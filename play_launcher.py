# play_launcher.py
# Öffnet dein lokales Play-Dashboard (HTML) im Standard-Browser.
# Enthält:
#  - PyInstaller-freundliche Pfadauflösung (resource_path)
#  - Kalibrierungsprüfung (H_world2img vorhanden? -> Button enable/disable)
#  - Optionales Routing per Hash-Fragment (z. B. #/setup?mode=x01)

from __future__ import annotations
from pathlib import Path
import sys
import json
import webbrowser

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox


# ---------- Pfadauflösung (funktioniert auch in PyInstaller-EXE) ----------
def resource_path(*parts: str) -> Path:
    """
    Liefert einen absoluten Pfad zu mitgelieferten Dateien.
    Unter PyInstaller zeigt _MEIPASS auf das entpackte Temp-Verzeichnis.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


# ---------- Kalibrierungsprüfung ----------
def is_calibration_valid() -> bool:
    """
    Gültig, wenn in kalibrierung.json eine 3x3-Matrix H_world2img vorhanden ist.
    (Für Demo/Entwicklung reicht die Identity-Matrix.)
    """
    cfg_path = resource_path("kalibrierung.json")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        H = cfg.get("H_world2img")
        return (
            isinstance(H, list)
            and len(H) == 3
            and all(isinstance(r, list) and len(r) == 3 for r in H)
        )
    except Exception:
        return False


# ---------- HTML im Browser öffnen ----------
def open_play_dashboard(mode: str | None = None, route: str = "setup") -> None:
    """
    Öffnet assets/play_dashboard.html im Standard-Browser.
    Optional: route + mode als Hash-Fragment (#/setup?mode=x01)
    """
    html = resource_path("assets", "play_dashboard.html")
    if not html.exists():
        QMessageBox.critical(None, "Fehlende Datei", f"HTML nicht gefunden:\n{html}")
        return

    url = QUrl.fromLocalFile(str(html))
    if mode:
        url.setFragment(
            f"/{route}?mode={mode}"
        )  # -> file:///.../play_dashboard.html#/setup?mode=x01

    # Primär QDesktopServices; Fallback auf webbrowser
    opened = QDesktopServices.openUrl(url)
    if not opened:
        webbrowser.open(url.toString())


# ---------- Minimal-Demo-Fenster (kannst du in dein MainWindow integrieren) ----------
class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TripleOne – Play")
        self.resize(380, 140)

        self.btn_play = QPushButton("Play – Spiele anzeigen")
        self.btn_play.clicked.connect(self.on_play_clicked)

        lay = QVBoxLayout(self)
        lay.addWidget(self.btn_play)

        # Button anhand der Kalibrierung aktivieren/deaktivieren
        self.btn_play.setEnabled(is_calibration_valid())

    def on_play_clicked(self):
        if not is_calibration_valid():
            # Hinweis – wenn du trotz fehlender Kalibrierung öffnen willst, kommentiere das 'return' aus.
            QMessageBox.warning(
                self, "Kalibrierung fehlt", "Bitte zuerst die Kalibrierung abschließen."
            )
            # return
        open_play_dashboard()  # Spiele-Grid
        # open_play_dashboard("x01")   # Beispiel: direkt in X01-Setup springen


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Demo()
    w.show()
    sys.exit(app.exec())
