"""
MainWindow - desktop UI for live sign-language subtitles.

Threading model:
  CaptureThread   -> InferenceThread.receive_frame()
  InferenceThread -> MainWindow._on_result()        (display)
  InferenceThread -> MainWindow._on_result_to_vcam() -> VirtualCamThread.push_frame()

Virtual camera output runs on a dedicated worker so pyvirtualcam blocking calls
never stall inference or the UI.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.startup_checks import can_launch, run_all_checks
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
    QCheckBox#vcam_toggle { color: #a0a0c0; font-size: 13px; spacing: 8px; }
    QCheckBox#vcam_toggle::indicator {
        width: 16px; height: 16px; border-radius: 4px;
        border: 1px solid #3a3a5c; background: #1e1e35;
    }
    QCheckBox#vcam_toggle::indicator:checked { background: #7c83fd; border-color: #7c83fd; }
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

        self._capture_thread = CaptureThread(camera_index=self._camera_index)
        self._inference_thread = InferenceThread()
        self._vcam_thread = VirtualCamThread(fps=20)

        self._wire_threads()
        self._build_ui()
        self.statusBar().showMessage("Ready - click Start to begin")

    def _wire_threads(self):
        self._capture_thread.frame_ready.connect(self._inference_thread.receive_frame)
        self._capture_thread.fps_updated.connect(self._on_fps_update)
        self._capture_thread.error_occurred.connect(self._on_runtime_error)

        self._inference_thread.result_ready.connect(self._on_result)
        self._inference_thread.result_ready.connect(self._on_result_to_vcam)
        self._inference_thread.status_updated.connect(self._on_status_update)
        self._inference_thread.error_occurred.connect(self._on_runtime_error)
        self._inference_thread.performance_updated.connect(self._on_inference_perf_update)

        self._vcam_thread.status_changed.connect(self._on_vcam_status)
        self._vcam_thread.error_occurred.connect(self._on_vcam_error)

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

        self._feed_label = QLabel("Camera feed will appear here\n\nClick  Start  to begin")
        self._feed_label.setObjectName("feed_placeholder")
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(self._feed_label)
        return panel

    def _build_control_panel(self):
        panel = QWidget()
        panel.setFixedWidth(230)
        panel.setStyleSheet("background-color: #0e0e1f; border-left: 1px solid #1e1e35;")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(12)

        title = QLabel("GESTURA")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Sign Language Subtitles")
        sub.setObjectName("subtitle_label")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        layout.addWidget(_divider())

        status_row = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #e74c3c; font-size: 18px;")
        self._status_text = QLabel("Idle")
        status_row.addWidget(self._status_dot)
        status_row.addWidget(self._status_text)
        status_row.addStretch()
        layout.addLayout(status_row)

        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("start_btn")
        self._start_btn.clicked.connect(self._on_start)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)

        self._clear_btn = QPushButton("Clear Text")
        self._clear_btn.setObjectName("clear_btn")
        self._clear_btn.clicked.connect(self._on_clear)

        layout.addWidget(self._start_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._clear_btn)

        layout.addWidget(_divider())

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

        self._vcam_checkbox = QCheckBox("Virtual Camera (Meetings)")
        self._vcam_checkbox.setObjectName("vcam_toggle")
        self._vcam_checkbox.setChecked(False)
        self._vcam_checkbox.stateChanged.connect(self._on_vcam_toggle)
        layout.addWidget(self._vcam_checkbox)

        self._vcam_status_label = QLabel("Virtual cam: off")
        self._vcam_status_label.setStyleSheet("color: #444466; font-size: 11px;")
        layout.addWidget(self._vcam_status_label)

        layout.addWidget(_divider())

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
        return panel

    @pyqtSlot(object, str)
    def _on_result(self, frame: np.ndarray, sentence: str):
        frame = self._draw_subtitle(frame, sentence)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self._feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed_label.setPixmap(pixmap)

    @pyqtSlot(object, str)
    def _on_result_to_vcam(self, frame: np.ndarray, sentence: str):
        if self._vcam_enabled and self._vcam_thread.isRunning():
            vcam_frame = self._draw_subtitle(frame.copy(), sentence)
            self._vcam_thread.push_frame(vcam_frame)

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

    @pyqtSlot(bool)
    def _on_vcam_status(self, active: bool):
        if active:
            self._vcam_status_label.setText("Virtual cam: active")
            self._vcam_status_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
            self.statusBar().showMessage(
                "Virtual camera active - select it in your meeting app."
            )
        else:
            self._vcam_status_label.setText("Virtual cam: off")
            self._vcam_status_label.setStyleSheet("color: #444466; font-size: 11px;")

    @pyqtSlot(str)
    def _on_vcam_error(self, message: str):
        self._vcam_status_label.setText("Virtual cam: error")
        self._vcam_status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
        self.statusBar().showMessage(f"Virtual cam error: {message}")

    def _on_start(self):
        if self._is_running:
            return

        checks = run_all_checks(camera_index=self._camera_index)
        if not can_launch(checks):
            for check in checks:
                if check["required"] and not check["ok"]:
                    self._show_error_dialog(
                        title=f"Cannot start - {check['name']}",
                        message=check["message"],
                    )
                    return

        for check in checks:
            if not check["required"] and not check["ok"]:
                self.statusBar().showMessage(f"Warning: {check['message']}")

        self._inference_thread.start()
        self._capture_thread.start()
        if self._vcam_enabled and not self._vcam_thread.isRunning():
            self._vcam_thread.start()

        self._is_running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._cam_spinbox.setEnabled(False)
        self._status_dot.setStyleSheet("color: #2ecc71; font-size: 18px;")
        self._status_text.setText("Running")
        self.statusBar().showMessage("Camera started - show a sign to begin")

    def _on_stop(self):
        if not self._is_running:
            return

        self.statusBar().showMessage("Stopping...")

        self._capture_thread.stop()
        self._inference_thread.stop()
        if self._vcam_thread.isRunning():
            self._vcam_thread.stop()

        self._is_running = False
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
        self.statusBar().showMessage("Stopped.")

    def _on_clear(self):
        self._inference_thread.clear()
        self.statusBar().showMessage("Text cleared.")

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
