"""
InferenceThread - MediaPipe + model inference worker.

The capture thread can produce frames faster than inference can consume them.
This worker keeps only the freshest frame, so the UI stays close to real time
instead of lagging behind a growing queue.
"""

from collections import deque
from pathlib import Path
from threading import Lock
import time
import traceback

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from core.smoother import TemporalSmoother


PHASE1_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE1_DIR.parent
MODEL_CANDIDATES = [
    PHASE1_DIR / "best_gesture_model_wlasl300_inference.pt",
    PROJECT_ROOT / "MODEL_Training" / "Model" / "best_gesture_model_wlasl300_inference.pt",
    PHASE1_DIR / "best_gesture_model_wlasl300_epoch471.pt",
    PROJECT_ROOT / "MODEL_Training" / "Model" / "best_gesture_model_wlasl300_epoch471.pt",
]

SEQUENCE_LENGTH = 60
MIN_SEQUENCE_FOR_PREDICTION = 20
DEFAULT_THRESHOLD = 0.40
SMOOTHING_WINDOW = 6
SMOOTHING_MIN_VOTES = 3
INFERENCE_FRAME_SIZE = (640, 360)


class InferenceThread(QThread):
    result_ready = pyqtSignal(object, str)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    performance_updated = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self._running = False
        self._assets = None
        self._actions = None
        self._model = None
        self._device = None
        self._model_path = None
        self._holistic = None

        self._frame_lock = Lock()
        self._state_lock = Lock()
        self._frame_queue = deque(maxlen=1)
        self._sequence = deque(maxlen=SEQUENCE_LENGTH)
        self._sentence = []
        self._smoother = TemporalSmoother(window=SMOOTHING_WINDOW, min_votes=SMOOTHING_MIN_VOTES)
        self._threshold = DEFAULT_THRESHOLD
        self._session_version = 0
        self._last_model_ms = 0.0

    @pyqtSlot(object)
    def receive_frame(self, frame):
        """Accept the newest frame and discard anything stale."""
        with self._frame_lock:
            self._frame_queue.clear()
            self._frame_queue.append(frame)

    def run(self):
        """Run MediaPipe and model prediction in the background thread."""
        self._running = True
        self._reset_session_state()

        try:
            self._ensure_model_loaded()
            self._holistic = self._assets.mp_holistic.Holistic(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            perf_started_at = time.perf_counter()
            perf_frames = 0
            mediapipe_ms_total = 0.0
            model_ms_total = 0.0

            while self._running and not self.isInterruptionRequested():
                frame = self._pop_latest_frame()
                if frame is None:
                    self.msleep(5)
                    continue

                frame = cv2.resize(frame, INFERENCE_FRAME_SIZE, interpolation=cv2.INTER_AREA)

                mediapipe_started_at = time.perf_counter()
                annotated_image, results = self._assets.mediapipe_detection(
                    frame,
                    self._holistic,
                )
                self._assets.draw_landmark(annotated_image, results)
                mediapipe_ms = (time.perf_counter() - mediapipe_started_at) * 1000.0

                keypoints = self._assets.extract_keypoints(results)
                sentence_text = self._predict_sentence(keypoints)

                perf_frames += 1
                mediapipe_ms_total += mediapipe_ms
                model_ms_total += self._last_model_ms
                perf_elapsed = time.perf_counter() - perf_started_at
                if perf_elapsed >= 1.0:
                    self.performance_updated.emit(
                        perf_frames / perf_elapsed,
                        mediapipe_ms_total / perf_frames,
                        model_ms_total / perf_frames,
                    )
                    perf_started_at = time.perf_counter()
                    perf_frames = 0
                    mediapipe_ms_total = 0.0
                    model_ms_total = 0.0

                self.result_ready.emit(annotated_image, sentence_text)
        except Exception as exc:
            traceback.print_exc()
            self.error_occurred.emit(f"Inference stopped: {exc}")
        finally:
            self._running = False
            self._clear_frame_queue()
            if self._holistic is not None:
                self._holistic.close()
                self._holistic = None
            print("[InferenceThread] Stopped.")

    def clear(self):
        """Reset the currently displayed sentence and temporal buffers."""
        self._clear_frame_queue()
        self._reset_session_state()

    def set_threshold(self, value: float):
        """Update confidence threshold from the UI slider."""
        with self._state_lock:
            self._threshold = value

    def stop(self, wait_ms: int = 3000) -> bool:
        """Request a clean stop and wait briefly for inference to finish."""
        self._running = False

        if not self.isRunning():
            return True

        self.requestInterruption()

        if not self.wait(wait_ms):
            message = "Inference thread did not stop within the timeout."
            print(f"[InferenceThread] WARNING: {message}")
            self.error_occurred.emit(message)
            return False

        return True

    def _ensure_model_loaded(self):
        if self._model is not None:
            return

        self._model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
        if self._model_path is None:
            expected = "\n".join(f"  - {path}" for path in MODEL_CANDIDATES)
            raise FileNotFoundError(
                f"Model file not found. Expected one of:\n{expected}"
            )

        self.status_updated.emit("Loading model...")
        self._load_assets()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self._model_path, map_location=self._device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        num_classes = int(checkpoint.get("num_classes", self._actions.shape[0])) if isinstance(checkpoint, dict) else int(self._actions.shape[0])
        if num_classes != int(self._actions.shape[0]):
            raise ValueError(
                f"Label mismatch: checkpoint has {num_classes} classes, but actions has {self._actions.shape[0]}"
            )

        self._model = self._assets.build_gesture_model(
            num_classes=num_classes,
            d_model=int(checkpoint.get("d_model", 256)) if isinstance(checkpoint, dict) else 256,
            nhead=int(checkpoint.get("nhead", 8)) if isinstance(checkpoint, dict) else 8,
            num_layers=int(checkpoint.get("num_layers", 4)) if isinstance(checkpoint, dict) else 4,
            dim_ff=int(checkpoint.get("dim_ff", 512)) if isinstance(checkpoint, dict) else 512,
            dropout=float(checkpoint.get("dropout", 0.0)) if isinstance(checkpoint, dict) else 0.0,
            seq_length=SEQUENCE_LENGTH,
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device)
        self._model.eval()
        self.status_updated.emit(f"Model loaded: {self._model_path.name}")
        print(f"[InferenceThread] PyTorch checkpoint loaded: {self._model_path.name}")

    def _load_assets(self):
        if self._assets is not None:
            return

        from core import inference_assets

        self._assets = inference_assets
        self._actions = inference_assets.actions

    def _pop_latest_frame(self):
        with self._frame_lock:
            if not self._frame_queue:
                return None
            frame = self._frame_queue.pop()
            self._frame_queue.clear()
            return frame

    def _clear_frame_queue(self):
        with self._frame_lock:
            self._frame_queue.clear()

    def _reset_session_state(self):
        with self._state_lock:
            self._sequence.clear()
            self._sentence = []
            self._smoother.reset()
            self._session_version += 1

    def _predict_sentence(self, keypoints) -> str:
        with self._state_lock:
            self._sequence.append(keypoints)
            sequence_ready = len(self._sequence) >= MIN_SEQUENCE_FOR_PREDICTION
            current_sentence = " ".join(self._sentence)
            threshold = self._threshold
            if sequence_ready:
                sequence_snapshot = list(self._sequence)
                if len(sequence_snapshot) < SEQUENCE_LENGTH:
                    pad = [sequence_snapshot[-1]] * (SEQUENCE_LENGTH - len(sequence_snapshot))
                    sequence_snapshot = pad + sequence_snapshot
                elif len(sequence_snapshot) > SEQUENCE_LENGTH:
                    sequence_snapshot = sequence_snapshot[-SEQUENCE_LENGTH:]
            else:
                sequence_snapshot = None
            session_version = self._session_version

        if not sequence_ready:
            self._last_model_ms = 0.0
            return current_sentence

        input_data = np.expand_dims(np.array(sequence_snapshot, dtype=np.float32), axis=0)
        input_tensor = torch.from_numpy(input_data).to(self._device)

        model_started_at = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(input_tensor)
            result = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self._last_model_ms = (time.perf_counter() - model_started_at) * 1000.0

        predicted_index = int(np.argmax(result))
        predicted_confidence = float(result[predicted_index])

        with self._state_lock:
            if session_version != self._session_version:
                return " ".join(self._sentence)

            stable_prediction = self._smoother.update(
                predicted_index,
                predicted_confidence,
                threshold,
            )

            if stable_prediction is not None:
                predicted_word = self._actions[stable_prediction]
                if not self._sentence or self._sentence[-1] != predicted_word:
                    self._sentence.append(predicted_word)

            if len(self._sentence) > 5:
                self._sentence = self._sentence[-5:]

            return " ".join(self._sentence)
