# login_dialog.py
from PyQt6 import QtWidgets, QtCore


class LoginDialog(QtWidgets.QDialog):
    def __init__(self, auth_client, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Account")
        self.auth = auth_client
        tabs = QtWidgets.QTabWidget()

        # --- Login Tab ---
        w_login = QtWidgets.QWidget()
        l1 = QtWidgets.QFormLayout(w_login)
        self.le_email = QtWidgets.QLineEdit()
        self.le_email.setPlaceholderText("email@example.com")
        self.le_pass = QtWidgets.QLineEdit()
        self.le_pass.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        l1.addRow("E-Mail", self.le_email)
        l1.addRow("Passwort", self.le_pass)
        self.btn_login = QtWidgets.QPushButton("Login")
        l1.addRow(self.btn_login)

        # --- Register Tab ---
        w_reg = QtWidgets.QWidget()
        l2 = QtWidgets.QFormLayout(w_reg)
        self.re_email = QtWidgets.QLineEdit()
        self.re_pass = QtWidgets.QLineEdit()
        self.re_pass.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.re_name = QtWidgets.QLineEdit()
        l2.addRow("E-Mail", self.re_email)
        l2.addRow("Passwort (min. 6)", self.re_pass)
        l2.addRow("Anzeigename", self.re_name)
        self.btn_reg = QtWidgets.QPushButton("Registrieren")
        l2.addRow(self.btn_reg)

        tabs.addTab(w_login, "Login")
        tabs.addTab(w_reg, "Registrieren")

        btn_close = QtWidgets.QPushButton("Schließen")

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(tabs)
        root.addWidget(btn_close, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # wiring
        self.btn_login.clicked.connect(self._do_login)
        self.btn_reg.clicked.connect(self._do_register)
        btn_close.clicked.connect(self.reject)

        self.result_payload = None

    def _alert(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Account", msg)

    def _do_login(self):
        email = self.le_email.text().strip()
        pw = self.le_pass.text()
        if not email or not pw:
            self._alert("Bitte E-Mail und Passwort eingeben.")
            return
        try:
            js = self.auth.login(email, pw)
            self.result_payload = js
            QtWidgets.QMessageBox.information(
                self,
                "Login",
                f"Willkommen {js['user'].get('displayName') or js['user']['email']}",
            )
            self.accept()
        except Exception as e:
            self._alert(f"Login fehlgeschlagen:\n{e}")

    def _do_register(self):
        email = self.re_email.text().strip()
        pw = self.re_pass.text()
        name = self.re_name.text().strip()
        if not email or not pw:
            self._alert("Bitte E-Mail und Passwort eingeben.")
            return
        try:
            self.auth.register(email, pw, name or None)
            QtWidgets.QMessageBox.information(
                self, "Registrierung", "Konto angelegt. Du kannst dich jetzt einloggen."
            )
        except Exception as e:
            self._alert(f"Registrierung fehlgeschlagen:\n{e}")
