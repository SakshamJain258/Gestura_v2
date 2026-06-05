"""
VirtualCamThread - dedicated virtual camera output worker.

pyvirtualcam.Camera.send() can block waiting for the consumer app. This worker
isolates that blocking call so inference and UI remain responsive.
"""

import time
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


class VirtualCamThread(QThread):
    """Push annotated frames to a virtual camera on a separate thread."""

    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(bool)

    def __init__(self, fps: int = 20):
        super().__init__()
        self._fps = fps
        self._running = False
        self._frame_queue = deque(maxlen=2)
        self._width = None
        self._height = None

    def push_frame(self, frame: np.ndarray):
        """Queue the newest frame for virtual cam output without blocking."""
        if not self.isRunning():
            return

        if self._width is None:
            self._height, self._width = frame.shape[:2]

        self._frame_queue.append(frame)

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        if not PYVIRTUALCAM_AVAILABLE:
            self.error_occurred.emit("pyvirtualcam not installed. Virtual camera unavailable.")
            return

        self._running = True

        timeout = 5.0
        start = time.time()
        while self._width is None:
            if time.time() - start > timeout:
                self.error_occurred.emit("VirtualCamThread timed out waiting for first frame.")
                return
            time.sleep(0.05)

        try:
            with pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                fmt=PixelFormat.BGR,
                print_fps=False,
            ) as cam:
                print(
                    f"[VirtualCamThread] Virtual camera started: "
                    f"{self._width}x{self._height} @ {self._fps}fps -> {cam.device}"
                )
                self.status_changed.emit(True)

                frame_interval = 1.0 / self._fps
                last_frame_time = time.time()

                while self._running:
                    now = time.time()
                    if now - last_frame_time < frame_interval:
                        time.sleep(0.001)
                        continue

                    if not self._frame_queue:
                        time.sleep(0.001)
                        continue

                    frame = self._frame_queue.pop()

                    if frame.shape[1] != self._width or frame.shape[0] != self._height:
                        frame = cv2.resize(frame, (self._width, self._height))

                    cam.send(frame)
                    last_frame_time = now

        except Exception as exc:
            err = f"Virtual camera error: {exc}"
            print(f"[VirtualCamThread] {err}")
            self.error_occurred.emit(err)
        finally:
            self._running = False
            self._frame_queue.clear()
            self._width = None
            self._height = None
            self.status_changed.emit(False)
            print("[VirtualCamThread] Virtual camera stopped.")
