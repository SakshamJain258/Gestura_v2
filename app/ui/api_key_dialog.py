"""
ApiKeyDialog — UI dialog for configuring the Gemini API key.

Provides:
  - API key input field (masked)
  - Save button
  - Test button (validates key with a live ping)
  - Status display

Usage:
    dialog = ApiKeyDialog(parent=main_window)
    dialog.exec()
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from core.llm_client import load_api_key, save_api_key, test_api_key


DIALOG_STYLE = """
QDialog {
    background-color: #12121f;
    color: #e0e0f0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QLabel { color: #e0e0f0; }
QLabel#title_label {
    font-size: 16px;
    font-weight: bold;
    color: #7c83fd;
}
QLabel#hint_label {
    color: #888899;
    font-size: 11px;
}
QLineEdit {
    background-color: #1e1e35;
    color: #e0e0f0;
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}
QLineEdit:focus { border-color: #7c83fd; }
QPushButton {
    background-color: #1e1e35;
    color: #e0e0f0;
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}
QPushButton:hover { background-color: #2e2e50; border-color: #7c83fd; }
QPushButton:pressed { background-color: #7c83fd; color: #12121f; }
QPushButton#save_btn { background-color: #1a3a2a; border-color: #2ecc71; color: #2ecc71; }
QPushButton#save_btn:hover { background-color: #2ecc71; color: #12121f; }
QPushButton#test_btn { background-color: #2a2a1a; border-color: #f39c12; color: #f39c12; }
QPushButton#test_btn:hover { background-color: #f39c12; color: #12121f; }
QLabel#status_ok { color: #2ecc71; font-size: 12px; }
QLabel#status_err { color: #e74c3c; font-size: 12px; }
QLabel#status_info { color: #7c83fd; font-size: 12px; }
"""


class TestKeyThread(QThread):
    """Background thread to test API key without blocking the UI."""
    result = pyqtSignal(bool, str)

    def __init__(self, api_key: str, parent=None):
        super().__init__(parent)
        self._api_key = api_key

    def run(self):
        ok, msg = test_api_key(self._api_key)
        self.result.emit(ok, msg)


class ApiKeyDialog(QDialog):
    """Dialog for configuring and testing the Gemini API key."""

    key_saved = pyqtSignal(str)   # Emits the new key when saved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gemini API Key — Settings")
        self.setMinimumWidth(480)
        self.setStyleSheet(DIALOG_STYLE)
        self.setModal(True)

        self._test_thread: TestKeyThread | None = None
        self._build_ui()
        self._load_existing_key()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Gemini API Key")
        title.setObjectName("title_label")
        layout.addWidget(title)

        hint = QLabel(
            "Get your free API key at "
            "<a href='https://aistudio.google.com/app/apikey' "
            "style='color: #7c83fd;'>aistudio.google.com</a>."
        )
        hint.setObjectName("hint_label")
        hint.setOpenExternalLinks(True)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("AIza...")
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self._key_input)

        show_row = QHBoxLayout()
        self._show_toggle = QPushButton("Show")
        self._show_toggle.setFixedWidth(70)
        self._show_toggle.setCheckable(True)
        self._show_toggle.toggled.connect(self._toggle_visibility)
        show_row.addStretch()
        show_row.addWidget(self._show_toggle)
        layout.addLayout(show_row)

        btn_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Key")
        self._test_btn.setObjectName("test_btn")
        self._test_btn.clicked.connect(self._on_test)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("save_btn")
        self._save_btn.clicked.connect(self._on_save)

        btn_row.addWidget(self._test_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setObjectName("status_info")
        layout.addWidget(self._status_label)

        layout.addStretch()

        close_btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn.rejected.connect(self.reject)
        layout.addWidget(close_btn)

    def _load_existing_key(self):
        key = load_api_key()
        if key:
            self._key_input.setText(key)
            self._set_status("Key loaded from config.", "info")

    def _toggle_visibility(self, checked: bool):
        self._key_input.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self._show_toggle.setText("Hide" if checked else "Show")

    def _on_test(self):
        key = self._key_input.text().strip()
        if not key:
            self._set_status("Please enter an API key first.", "err")
            return

        self._test_btn.setEnabled(False)
        self._set_status("Testing key...", "info")

        self._test_thread = TestKeyThread(key, parent=self)
        self._test_thread.result.connect(self._on_test_result)
        self._test_thread.start()

    def _on_test_result(self, ok: bool, message: str):
        self._test_btn.setEnabled(True)
        self._set_status(message, "ok" if ok else "err")

    def _on_save(self):
        key = self._key_input.text().strip()
        if not key:
            self._set_status("Cannot save an empty key.", "err")
            return
        save_api_key(key)
        self._set_status("Key saved successfully.", "ok")
        self.key_saved.emit(key)

    def _set_status(self, message: str, level: str = "info"):
        obj_names = {"ok": "status_ok", "err": "status_err", "info": "status_info"}
        self._status_label.setObjectName(obj_names.get(level, "status_info"))
        self._status_label.setText(message)
        # Force style refresh
        self._status_label.setStyle(self._status_label.style())
