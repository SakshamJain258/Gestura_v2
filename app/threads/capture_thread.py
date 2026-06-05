"""
CaptureThread - webcam capture worker.

This thread has one job: read camera frames and emit them quickly. It does not
run MediaPipe or model inference, so the camera stays responsive even when
inference is slower than real time.
"""

import time

import cv2
from PyQt6.QtCore import QThread, pyqtSignal


class CaptureThread(QThread):
    frame_ready = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self._running = False

    def run(self):
        """Read frames in the background thread until stopped."""
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            message = f"Could not open camera {self.camera_index}."
            print(f"[CaptureThread] ERROR: {message}")
            self.error_occurred.emit(message)
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 25)

        prev_time = time.time()
        frame_count = 0
        consecutive_failures = 0

        try:
            while self._running and not self.isInterruptionRequested():
                ret, frame = cap.read()

                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 60:
                        message = "Camera stopped returning frames."
                        print(f"[CaptureThread] ERROR: {message}")
                        self.error_occurred.emit(message)
                        break
                    self.msleep(20)
                    continue

                consecutive_failures = 0
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame)

                frame_count += 1
                now = time.time()
                elapsed = now - prev_time
                if elapsed >= 1.0:
                    self.fps_updated.emit(frame_count / elapsed)
                    frame_count = 0
                    prev_time = now
        finally:
            self._running = False
            cap.release()
            print("[CaptureThread] Camera released.")

    def stop(self, wait_ms: int = 2000) -> bool:
        """Request a clean stop and wait briefly for the camera to release."""
        self._running = False

        if not self.isRunning():
            return True

        self.requestInterruption()

        if not self.wait(wait_ms):
            message = "Camera thread did not stop within the timeout."
            print(f"[CaptureThread] WARNING: {message}")
            self.error_occurred.emit(message)
            return False

        return True
