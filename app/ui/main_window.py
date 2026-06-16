"""
MainWindow - desktop UI for live sign-language subtitles.

Threading model:
  CaptureThread   -> InferenceThread.receive_frame()
  InferenceThread -> MainWindow._on_result()        (display)
  InferenceThread -> MainWindow._on_result_to_vcam() -> VirtualCamThread.push_frame()
  InferenceThread -> word_detected -> CaptionBuffer.append()
  CaptionBuffer   -> signing_paused -> PauseDetector -> GeminiClient
  GeminiClient    -> caption_refined -> UI llm_label

Task 3:  CaptionBuffer integrated for sentence state
Task 7:  Mode toggle (Word / Fingerspell) via ModeController
Task 8:  Settings button → ApiKeyDialog
Task 9:  LLM refined caption display
Task 10: Full PyQt6 UI polish
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.startup_checks import can_launch, run_all_checks
from core.caption_buffer import CaptionBuffer
from core.mode_controller import ModeController, MODE_WORD, MODE_FINGER
from core.llm_client import GeminiClient, load_api_key
from core.pause_detector import PauseDetector
from threads.capture_thread import CaptureThread
from threads.inference_thread import InferenceThread
from threads.virtual_cam_thread import VirtualCamThread


DARK_STYLE = """
    QMainWindow, QWidget {
        background-color: #12121f;
        color: #e0e0f0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 14px;
    }
    QLabel#title {
        font-size: 22px;
        font-weight: bold;
        color: #7c83fd;
        letter-spacing: 2px;
    }
    QLabel#subtitle_label { font-size: 13px; color: #888899; }
    QPushButton {
        background-color: #1e1e35;
        color: #e0e0f0;
        border: 1px solid #3a3a5c;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 14px;
    }
    QPushButton:hover { background-color: #2e2e50; border-color: #7c83fd; }
    QPushButton:pressed { background-color: #7c83fd; color: #12121f; }
    QPushButton:disabled { background-color: #1a1a2a; color: #444466; border-color: #2a2a40; }
    QPushButton#start_btn { background-color: #1a3a2a; border-color: #2ecc71; color: #2ecc71; }
    QPushButton#start_btn:hover { background-color: #2ecc71; color: #12121f; }
    QPushButton#stop_btn { background-color: #3a1a1a; border-color: #e74c3c; color: #e74c3c; }
    QPushButton#stop_btn:hover { background-color: #e74c3c; color: #12121f; }
    QPushButton#clear_btn { background-color: #2a2a1a; border-color: #f39c12; color: #f39c12; }
    QPushButton#clear_btn:hover { background-color: #f39c12; color: #12121f; }
    QPushButton#mode_btn { background-color: #1a2a3a; border-color: #3498db; color: #3498db; font-size: 12px; padding: 7px 12px; }
    QPushButton#mode_btn:hover { background-color: #3498db; color: #12121f; }
    QPushButton#settings_btn { background-color: #1e1e35; border-color: #666688; color: #888899; font-size: 11px; padding: 5px 10px; }
    QPushButton#settings_btn:hover { background-color: #2e2e50; color: #e0e0f0; border-color: #7c83fd; }
    QPushButton#gemini_toggle { background-color: #1a2a1a; border-color: #2ecc71; color: #2ecc71; font-size: 11px; padding: 5px 10px; }
    QPushButton#gemini_toggle:checked { background-color: #2ecc71; color: #12121f; }
    QPushButton#gemini_toggle:hover { background-color: #27ae60; color: #12121f; }
    QCheckBox { color: #a0a0c0; font-size: 13px; spacing: 8px; }
    QCheckBox::indicator {
        width: 16px; height: 16px; border-radius: 4px;
        border: 1px solid #3a3a5c; background: #1e1e35;
    }
    QCheckBox::indicator:checked { background: #7c83fd; border-color: #7c83fd; }
    QSlider::groove:horizontal { height: 6px; background: #2a2a45; border-radius: 3px; }
    QSlider::handle:horizontal {
        background: #7c83fd; width: 16px; height: 16px;
        margin: -5px 0; border-radius: 8px;
    }
    QSlider::sub-page:horizontal { background: #7c83fd; border-radius: 3px; }
    QFrame#divider { background-color: #2a2a45; max-height: 1px; }
    QLabel#feed_placeholder {
        background-color: #0a0a18; color: #3a3a5c;
        font-size: 16px; border: 2px dashed #2a2a45; border-radius: 12px;
    }
    QLabel#fps_label { color: #2ecc71; font-size: 13px; font-family: 'Consolas', monospace; }
    QSpinBox {
        background-color: #1e1e35; color: #e0e0f0;
        border: 1px solid #3a3a5c; border-radius: 6px; padding: 4px 8px;
    }
    QSpinBox::up-button, QSpinBox::down-button { background-color: #2a2a45; border: none; width: 18px; }
    QStatusBar { background-color: #0a0a18; color: #666680; font-size: 12px; }
    QLabel#llm_label {
        background-color: #0d1a2d;
        color: #7ec8e3;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        font-style: italic;
    }
    QLabel#llm_thinking {
        color: #7c83fd;
        font-size: 11px;
    }
    QLabel#mode_label {
        color: #3498db;
        font-size: 11px;
        font-weight: bold;
    }
    QLabel#gemini_status {
        color: #666688;
        font-size: 10px;
    }
"""


def _divider():
    line = QFrame()
    line.setObjectName("divider")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gestura")
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(DARK_STYLE)

        self._camera_index = 0
        self._is_running = False
        self._vcam_enabled = False

        # ── Core components ────────────────────────────────────────────────────
        self._capture_thread = CaptureThread(camera_index=self._camera_index)
        self._inference_thread = InferenceThread()
        self._vcam_thread = VirtualCamThread(fps=20)

        # Task 3: CaptionBuffer
        self._caption_buffer = CaptionBuffer(max_words=5, pause_threshold_sec=2.5)

        # Task 7: ModeController
        self._mode_controller = ModeController(self._inference_thread)

        # Task 8: Gemini client
        self._gemini_client = GeminiClient()
        self._gemini_enabled = False

        # Task 9: PauseDetector
        self._pause_detector = PauseDetector(self._caption_buffer, self._gemini_client)
        self._pause_detector.connect_all()

        self._wire_threads()
        self._build_ui()
        self.statusBar().showMessage("Ready — click Start to begin")

        # Check Gemini availability after UI is built
        self._update_gemini_status()

    # ── Wiring ────────────────────────────────────────────────────────────────

    def _wire_threads(self):
        self._capture_thread.frame_ready.connect(self._inference_thread.receive_frame)
        self._capture_thread.fps_updated.connect(self._on_fps_update)
        self._capture_thread.error_occurred.connect(self._on_runtime_error)

        self._inference_thread.result_ready.connect(self._on_result)
        self._inference_thread.result_ready.connect(self._on_result_to_vcam)
        self._inference_thread.status_updated.connect(self._on_status_update)
        self._inference_thread.error_occurred.connect(self._on_runtime_error)
        self._inference_thread.performance_updated.connect(self._on_inference_perf_update)

        # Task 3: word_detected → CaptionBuffer
        self._inference_thread.word_detected.connect(self._caption_buffer.append)
        self._caption_buffer.sentence_changed.connect(self._on_caption_changed)

        # Task 9: LLM refined caption
        self._pause_detector.refined_caption.connect(self._on_llm_caption)
        self._pause_detector.llm_active.connect(self._on_llm_active)

        self._vcam_thread.status_changed.connect(self._on_vcam_status)
        self._vcam_thread.error_occurred.connect(self._on_vcam_error)

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_feed_panel(), stretch=5)
        root.addWidget(self._build_control_panel(), stretch=1)

    def _build_feed_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 8, 16)
        layout.setSpacing(8)

        self._feed_label = QLabel("Camera feed will appear here\n\nClick  Start  to begin")
        self._feed_label.setObjectName("feed_placeholder")
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._feed_label, stretch=1)

        # Task 9: LLM refined caption strip at bottom
        self._llm_label = QLabel("Gemini refined caption will appear here when you pause signing...")
        self._llm_label.setObjectName("llm_label")
        self._llm_label.setWordWrap(True)
        self._llm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._llm_label.setVisible(False)
        layout.addWidget(self._llm_label)

        self._llm_thinking_label = QLabel("✦ Gemini is refining...")
        self._llm_thinking_label.setObjectName("llm_thinking")
        self._llm_thinking_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._llm_thinking_label.setVisible(False)
        layout.addWidget(self._llm_thinking_label)

        return panel

    def _build_control_panel(self):
        # Wrap control panel in a scroll area to prevent clipping/congestion on small screens
        scroll = QScrollArea()
        scroll.setFixedWidth(260)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background-color: #0e0e1f; border-left: 1px solid #1e1e35; }"
        )

        panel = QWidget()
        panel.setStyleSheet("background-color: #0e0e1f;")
        scroll.setWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        title = QLabel("GESTURA")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Sign Language Subtitles")
        sub.setObjectName("subtitle_label")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        layout.addWidget(_divider())

        # ── Status ────────────────────────────────────────────────────────────
        status_row = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #e74c3c; font-size: 18px;")
        self._status_text = QLabel("Idle")
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_text)
        status_row.addStretch()
        layout.addLayout(status_row)

        # ── Start / Stop / Clear ──────────────────────────────────────────────
        control_row = QHBoxLayout()
        self._start_btn = QPushButton("▶ Start")
        self._start_btn.setObjectName("start_btn")
        self._start_btn.clicked.connect(self._on_start)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        
        control_row.addWidget(self._start_btn)
        control_row.addWidget(self._stop_btn)
        layout.addLayout(control_row)

        self._clear_btn = QPushButton("✕  Clear Text")
        self._clear_btn.setObjectName("clear_btn")
        self._clear_btn.clicked.connect(self._on_clear)
        layout.addWidget(self._clear_btn)

        layout.addWidget(_divider())

        # ── Mode Toggle (Task 7) ──────────────────────────────────────────────
        layout.addWidget(QLabel("Recognition Mode"))
        self._mode_btn = QPushButton("⌨  Word-level ASL")
        self._mode_btn.setObjectName("mode_btn")
        self._mode_btn.setToolTip(
            "Toggle between word-level ASL recognition\nand letter-by-letter fingerspelling."
        )
        self._mode_btn.clicked.connect(self._on_mode_toggle)
        layout.addWidget(self._mode_btn)

        self._mode_label = QLabel("Layer 1: Word ASL active")
        self._mode_label.setObjectName("mode_label")
        layout.addWidget(self._mode_label)
        self._mode_controller.mode_changed.connect(self._on_mode_changed)

        layout.addWidget(_divider())

        # ── Camera Index ──────────────────────────────────────────────────────
        layout.addWidget(QLabel("Camera Index"))
        cam_row = QHBoxLayout()
        self._cam_spinbox = QSpinBox()
        self._cam_spinbox.setMinimum(0)
        self._cam_spinbox.setMaximum(5)
        self._cam_spinbox.setValue(self._camera_index)
        self._cam_spinbox.setToolTip(
            "0 = built-in webcam\n1 = external USB camera\nChange only when stopped."
        )
        self._cam_spinbox.valueChanged.connect(self._on_camera_index_change)
        cam_row.addWidget(self._cam_spinbox)
        cam_row.addStretch()
        layout.addLayout(cam_row)

        layout.addWidget(_divider())

        # ── Virtual Camera (Task 5) ───────────────────────────────────────────
        self._vcam_checkbox = QCheckBox("Virtual Camera (Meetings)")
        self._vcam_checkbox.setObjectName("vcam_toggle")
        self._vcam_checkbox.setChecked(False)
        self._vcam_checkbox.stateChanged.connect(self._on_vcam_toggle)
        layout.addWidget(self._vcam_checkbox)

        self._vcam_status_label = QLabel("Virtual cam: off")
        self._vcam_status_label.setStyleSheet("color: #444466; font-size: 11px;")
        layout.addWidget(self._vcam_status_label)

        # ── Show Landmarks ───────────────────────────────────────────────────
        self._landmarks_checkbox = QCheckBox("Show Landmarks Overlay")
        self._landmarks_checkbox.setChecked(True)
        self._landmarks_checkbox.stateChanged.connect(self._on_landmarks_toggle)
        layout.addWidget(self._landmarks_checkbox)

        layout.addWidget(_divider())

        # ── Confidence Threshold ──────────────────────────────────────────────
        layout.addWidget(QLabel("Confidence Threshold"))
        thresh_val_row = QHBoxLayout()
        thresh_val_row.addStretch()
        self._threshold_value_label = QLabel("40%")
        self._threshold_value_label.setStyleSheet("color: #7c83fd;")
        thresh_val_row.addWidget(self._threshold_value_label)
        layout.addLayout(thresh_val_row)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setMinimum(5)
        self._threshold_slider.setMaximum(80)
        self._threshold_slider.setValue(40)
        self._threshold_slider.valueChanged.connect(self._on_threshold_change)
        layout.addWidget(self._threshold_slider)

        layout.addWidget(_divider())

        # ── Gemini AI (Tasks 8/9) ─────────────────────────────────────────────
        gemini_header = QHBoxLayout()
        gemini_header.addWidget(QLabel("Gemini AI Captions"))
        gemini_header.addStretch()

        self._settings_btn = QPushButton("⚙ Key")
        self._settings_btn.setObjectName("settings_btn")
        self._settings_btn.setToolTip("Configure Gemini API key")
        self._settings_btn.clicked.connect(self._on_settings)
        gemini_header.addWidget(self._settings_btn)
        layout.addLayout(gemini_header)

        self._gemini_toggle_btn = QPushButton("Enable Gemini")
        self._gemini_toggle_btn.setObjectName("gemini_toggle")
        self._gemini_toggle_btn.setCheckable(True)
        self._gemini_toggle_btn.setChecked(False)
        self._gemini_toggle_btn.toggled.connect(self._on_gemini_toggle)
        layout.addWidget(self._gemini_toggle_btn)

        self._gemini_status_label = QLabel("No API key configured")
        self._gemini_status_label.setObjectName("gemini_status")
        layout.addWidget(self._gemini_status_label)

        layout.addWidget(_divider())

        # ── Performance ───────────────────────────────────────────────────────
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Camera FPS:"))
        self._fps_label = QLabel("-")
        self._fps_label.setObjectName("fps_label")
        fps_row.addWidget(self._fps_label)
        fps_row.addStretch()
        layout.addLayout(fps_row)

        inference_row = QHBoxLayout()
        inference_row.addWidget(QLabel("Inference:"))
        self._inference_fps_label = QLabel("-")
        self._inference_fps_label.setObjectName("fps_label")
        inference_row.addWidget(self._inference_fps_label)
        inference_row.addStretch()
        layout.addLayout(inference_row)

        self._perf_label = QLabel("MP - ms | model - ms")
        self._perf_label.setStyleSheet("color: #666680; font-size: 11px;")
        layout.addWidget(self._perf_label)

        layout.addStretch()
        return scroll

    # ── Slots: Frame Display ──────────────────────────────────────────────────

    @pyqtSlot(object, str)
    def _on_result(self, frame: np.ndarray, sentence: str):
        """Display annotated frame in the feed panel."""
        frame = self._draw_subtitle(frame, sentence)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self._feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._feed_label.setPixmap(pixmap)

    @pyqtSlot(object, str)
    def _on_result_to_vcam(self, frame: np.ndarray, sentence: str):
        if self._vcam_enabled and self._vcam_thread.isRunning():
            vcam_frame = self._draw_subtitle(frame.copy(), sentence)
            self._vcam_thread.push_frame(vcam_frame)

    # ── Slots: CaptionBuffer (Task 3) ─────────────────────────────────────────

    @pyqtSlot(str)
    def _on_caption_changed(self, sentence: str):
        """CaptionBuffer sentence changed — could update a dedicated caption label if desired."""
        pass  # The subtitle is drawn on the frame by _on_result via InferenceThread sentence

    # ── Slots: LLM (Task 9) ───────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_llm_caption(self, refined_text: str):
        self._llm_thinking_label.setVisible(False)
        self._llm_label.setText(f"✦ {refined_text}")
        self._llm_label.setVisible(True)

    @pyqtSlot(bool)
    def _on_llm_active(self, active: bool):
        self._llm_thinking_label.setVisible(active)
        if active:
            self._llm_label.setVisible(False)

    # ── Slots: Performance ────────────────────────────────────────────────────

    @pyqtSlot(float)
    def _on_fps_update(self, fps: float):
        self._fps_label.setText(f"{fps:.1f}")
        color = "#2ecc71" if fps >= 25 else "#f39c12" if fps >= 15 else "#e74c3c"
        self._fps_label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-family: 'Consolas', monospace;"
        )

    @pyqtSlot(float, float, float)
    def _on_inference_perf_update(self, fps: float, mediapipe_ms: float, model_ms: float):
        self._inference_fps_label.setText(f"{fps:.1f}")
        color = "#2ecc71" if fps >= 15 else "#f39c12" if fps >= 8 else "#e74c3c"
        self._inference_fps_label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-family: 'Consolas', monospace;"
        )
        self._perf_label.setText(f"MP {mediapipe_ms:.0f} ms | model {model_ms:.0f} ms")

    @pyqtSlot(str)
    def _on_status_update(self, message: str):
        self.statusBar().showMessage(message)

    @pyqtSlot(str)
    def _on_runtime_error(self, message: str):
        if self._is_running:
            self._on_stop()
        self._show_error_dialog("Runtime Error", message)

    # ── Slots: Virtual Camera ─────────────────────────────────────────────────

    @pyqtSlot(bool)
    def _on_vcam_status(self, active: bool):
        if active:
            self._vcam_status_label.setText("Virtual cam: active")
            self._vcam_status_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
            self.statusBar().showMessage(
                "Virtual camera active — select it in your meeting app."
            )
        else:
            self._vcam_status_label.setText("Virtual cam: off")
            self._vcam_status_label.setStyleSheet("color: #444466; font-size: 11px;")

    @pyqtSlot(str)
    def _on_vcam_error(self, message: str):
        self._vcam_status_label.setText("Virtual cam: error")
        self._vcam_status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
        self.statusBar().showMessage(f"Virtual cam error: {message}")

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _on_start(self):
        if self._is_running:
            return

        checks = run_all_checks(camera_index=self._camera_index)
        if not can_launch(checks):
            for check in checks:
                if check["required"] and not check["ok"]:
                    self._show_error_dialog(
                        title=f"Cannot start — {check['name']}",
                        message=check["message"],
                    )
                    return

        for check in checks:
            if not check["required"] and not check["ok"]:
                self.statusBar().showMessage(f"Warning: {check['message']}")

        self._caption_buffer.start()
        self._inference_thread.start()
        self._capture_thread.start()
        self._mode_controller.sync_to_thread()

        if self._vcam_enabled and not self._vcam_thread.isRunning():
            self._vcam_thread.start()

        self._is_running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._cam_spinbox.setEnabled(False)
        self._status_dot.setStyleSheet("color: #2ecc71; font-size: 18px;")
        self._status_text.setText("Running")
        self.statusBar().showMessage("Camera started — show a sign to begin")

    def _on_stop(self):
        if not self._is_running:
            return

        self._is_running = False
        self.statusBar().showMessage("Stopping...")

        self._caption_buffer.stop()
        self._capture_thread.stop()
        self._inference_thread.stop()
        if self._vcam_thread.isRunning():
            self._vcam_thread.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._cam_spinbox.setEnabled(True)
        self._status_dot.setStyleSheet("color: #e74c3c; font-size: 18px;")
        self._status_text.setText("Idle")
        self._feed_label.clear()
        self._feed_label.setText("Camera feed will appear here\n\nClick  Start  to begin")
        self._fps_label.setText("-")
        self._inference_fps_label.setText("-")
        self._perf_label.setText("MP - ms | model - ms")
        self._llm_label.setVisible(False)
        self._llm_thinking_label.setVisible(False)
        self.statusBar().showMessage("Stopped.")

    def _on_clear(self):
        self._inference_thread.clear()
        self._caption_buffer.clear()
        self._llm_label.setVisible(False)
        self._llm_thinking_label.setVisible(False)
        self.statusBar().showMessage("Text cleared.")

    # ── Controls ──────────────────────────────────────────────────────────────

    def _on_threshold_change(self, value: int):
        self._threshold_value_label.setText(f"{value}%")
        self._inference_thread.set_threshold(value / 100.0)

    def _on_camera_index_change(self, value: int):
        self._camera_index = value

        try:
            self._capture_thread.frame_ready.disconnect(self._inference_thread.receive_frame)
            self._capture_thread.fps_updated.disconnect(self._on_fps_update)
            self._capture_thread.error_occurred.disconnect(self._on_runtime_error)
        except (RuntimeError, TypeError):
            pass

        self._capture_thread = CaptureThread(camera_index=value)
        self._capture_thread.frame_ready.connect(self._inference_thread.receive_frame)
        self._capture_thread.fps_updated.connect(self._on_fps_update)
        self._capture_thread.error_occurred.connect(self._on_runtime_error)
        self.statusBar().showMessage(f"Camera index set to {value}.")

    def _on_vcam_toggle(self, state: int):
        self._vcam_enabled = bool(state)

        if not self._vcam_enabled and self._vcam_thread.isRunning():
            self._vcam_thread.stop()
            self.statusBar().showMessage("Virtual camera disabled.")
        elif self._vcam_enabled and self._is_running and not self._vcam_thread.isRunning():
            self._vcam_thread.start()
            self.statusBar().showMessage("Virtual camera enabled.")

    def _on_landmarks_toggle(self, state: int):
        self._inference_thread.set_show_landmarks(bool(state))
        self.statusBar().showMessage("Landmarks overlay enabled." if state else "Landmarks overlay disabled.")

    # ── Task 7: Mode Toggle ───────────────────────────────────────────────────

    def _on_mode_toggle(self):
        self._mode_controller.toggle()

    @pyqtSlot(str)
    def _on_mode_changed(self, label: str):
        is_word = "Word" in label
        icon = "⌨ " if not is_word else "🤟 "
        self._mode_btn.setText(f"{icon}{label}")
        layer = "Layer 1: Word ASL" if is_word else "Layer 2: Fingerspell"
        self._mode_label.setText(f"{layer} active")
        self._on_clear()   # reset caption on mode switch

    # ── Task 8: Settings / Gemini ─────────────────────────────────────────────

    def _on_settings(self):
        from ui.api_key_dialog import ApiKeyDialog
        dialog = ApiKeyDialog(parent=self)
        dialog.key_saved.connect(self._on_api_key_saved)
        dialog.exec()

    @pyqtSlot(str)
    def _on_api_key_saved(self, _key: str):
        self._update_gemini_status()
        self.statusBar().showMessage("Gemini API key saved.")

    def _on_gemini_toggle(self, checked: bool):
        self._gemini_enabled = checked
        self._gemini_client.set_enabled(checked)
        self._pause_detector.set_enabled(checked)
        self._gemini_toggle_btn.setText("Disable Gemini" if checked else "Enable Gemini")

        if checked and not self._gemini_client.is_available():
            self._gemini_toggle_btn.setChecked(False)
            self._show_error_dialog(
                "No API Key",
                "Please configure a Gemini API key in Settings first.",
            )
            return

        if checked:
            self._llm_label.setVisible(True)
            self._gemini_status_label.setText("Gemini active")
            self._gemini_status_label.setStyleSheet("color: #2ecc71; font-size: 10px;")
        else:
            self._llm_label.setVisible(False)
            self._llm_thinking_label.setVisible(False)
            self._gemini_status_label.setText("Gemini off")
            self._gemini_status_label.setStyleSheet("color: #444466; font-size: 10px;")

    def _update_gemini_status(self):
        key = load_api_key()
        if key:
            self._gemini_status_label.setText("API key configured ✓")
            self._gemini_status_label.setStyleSheet("color: #2ecc71; font-size: 10px;")
            self._gemini_toggle_btn.setEnabled(True)
        else:
            self._gemini_status_label.setText("No API key configured")
            self._gemini_status_label.setStyleSheet("color: #666688; font-size: 10px;")
            self._gemini_toggle_btn.setEnabled(False)

    # ── Frame Compositor (Task 3) ─────────────────────────────────────────────

    def _draw_subtitle(self, frame: np.ndarray, text: str) -> np.ndarray:
        if not text:
            return frame

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1.0, 2

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad_x, pad_y = 20, 12
        center_x = w // 2

        top_left = (center_x - text_w // 2 - pad_x, h - text_h - baseline - pad_y * 2 - 16)
        bottom_right = (center_x + text_w // 2 + pad_x, h - 16)

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        text_x = center_x - text_w // 2
        text_y = h - baseline - pad_y - 16
        cv2.putText(frame, text, (text_x, text_y), font, scale, (30, 30, 50), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    # ── Error Dialog ──────────────────────────────────────────────────────────

    def _show_error_dialog(self, title: str, message: str):
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setStyleSheet(
            """
            QMessageBox { background-color: #1e1e35; color: #e0e0f0; }
            QLabel { color: #e0e0f0; }
            QPushButton {
                background-color: #2e2e50; color: #e0e0f0;
                border: 1px solid #3a3a5c; border-radius: 6px; padding: 6px 16px;
            }
            """
        )
        dialog.exec()

    def closeEvent(self, event):
        if self._is_running:
            self._on_stop()
        elif self._vcam_thread.isRunning():
            self._vcam_thread.stop()
        event.accept()
