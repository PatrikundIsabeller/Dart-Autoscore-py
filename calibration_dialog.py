# calibration_dialog.py – einfacher Kalibrier-Dialog (Ringe + Boardgröße)
from PyQt6 import QtCore, QtWidgets
import requests, os

AGENT_URL = os.environ.get("TRIPLEONE_AGENT", "http://127.0.0.1:4700")

DEFAULT_RINGS = {
    "double_outer": 0.995,
    "double_inner": 0.940,
    "triple_outer": 0.620,
    "triple_inner": 0.540,
    "bull_outer":   0.060,
    "bull_inner":   0.015
}

class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setModal(True)
        self.resize(520, 420)

        self.form = QtWidgets.QFormLayout()
        self.form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Felder
        self.ed_diam = QtWidgets.QDoubleSpinBox()
        self.ed_diam.setRange(200.0, 600.0)
        self.ed_diam.setDecimals(2)
        self.ed_diam.setSuffix(" mm")

        self.ed = {}
        for key in ["double_outer","double_inner","triple_outer","triple_inner","bull_outer","bull_inner"]:
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.0, 1.2)
            sp.setDecimals(4)
            sp.setSingleStep(0.001)
            self.ed[key] = sp

        self.form.addRow("Board diameter", self.ed_diam)
        self.form.addRow("--- Rings (radius ratio) ---", QtWidgets.QLabel(""))
        for k, w in self.ed.items():
            self.form.addRow(k.replace("_"," "), w)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.save)
        btns.rejected.connect(self.reject)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(self.form)
        root.addStretch(1)
        root.addWidget(btns)

        self.load_current()

    def load_current(self):
        try:
            cfg = requests.get(AGENT_URL.rstrip("/") + "/calibration", timeout=1.5).json()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Read calibration failed:\n{e}")
            cfg = {}

        rings = cfg.get("rings") or DEFAULT_RINGS
        self.ed_diam.setValue(float(cfg.get("boardDiameterMm", 451.0)))
        for k, sp in self.ed.items():
            sp.setValue(float(rings.get(k, DEFAULT_RINGS[k])))

        # Speichere Roh-Config für späteres Zusammenführen (homographies/dist nicht verlieren)
        self._base_cfg = {
            "boardDiameterMm": float(cfg.get("boardDiameterMm", 451.0)),
            "homographies": cfg.get("homographies") or [],
            "dist": cfg.get("dist"),
            "rings": rings
        }

    def save(self):
        # neue Werte zusammenbauen
        rings = {k: self.ed[k].value() for k in self.ed}
        payload = {
            "boardDiameterMm": self.ed_diam.value(),
            "homographies": self._base_cfg.get("homographies", []),
            "dist": self._base_cfg.get("dist"),
            "rings": rings
        }
        try:
            r = requests.put(AGENT_URL.rstrip("/") + "/calibration", json=payload, timeout=2.0)
            r.raise_for_status()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Save failed:\n{e}")
            return
        self.accept()
