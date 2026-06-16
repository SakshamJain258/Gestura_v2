"""
InferenceThread - MediaPipe + model inference worker.

The capture thread can produce frames faster than inference can consume them.
This worker keeps only the freshest frame, so the UI stays close to real time
instead of lagging behind a growing queue.

Task 3: Uses CaptionBuffer for sentence state management.
Task 7: Supports mode switching (word-level ASL / fingerspelling).
"""

from collections import deque
from pathlib import Path
from threading import Lock
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from core.smoother import TemporalSmoother


def _get_base_dir() -> Path:
    """Base resource directory — supports both frozen exe and normal Python."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]   # Gestura v2/


_BASE = _get_base_dir()
TRAINING_RESULTS = _BASE / "training" / "Results"
PHASE1_DIR = _BASE / "app"

MODEL_CANDIDATES = [
    # Run 3 (current best)
    TRAINING_RESULTS / "Run_3" / "gesture_model_300_inference.pt",
    TRAINING_RESULTS / "Run_3" / "gesture_model_300_best.pt",
    # Legacy / fallback locations
    PHASE1_DIR / "best_gesture_model_wlasl300_inference.pt",
    PHASE1_DIR / "best_gesture_model_wlasl300_epoch471.pt",
]

# Fingerspelling model path
FINGER_MODEL_PATH = TRAINING_RESULTS / "Fingerspelling" / "fingerspelling_model.pt"


SEQUENCE_LENGTH = 60
MIN_SEQUENCE_FOR_PREDICTION = 20
DEFAULT_THRESHOLD = 0.40
SMOOTHING_WINDOW = 6
SMOOTHING_MIN_VOTES = 3
INFERENCE_FRAME_SIZE = (640, 360)

MODE_WORD = "word"
MODE_FINGER = "finger"


class InferenceThread(QThread):
    result_ready = pyqtSignal(object, str)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    performance_updated = pyqtSignal(float, float, float)
    word_detected = pyqtSignal(str)    # Task 9: for LLM trigger via CaptionBuffer

    def __init__(self):
        super().__init__()
        self._running = False
        self._assets = None
        self._actions = None
        self._model = None
        self._finger_model = None
        self._finger_actions = None
        self._device = None
        self._model_path = None
        self._holistic = None
        self._mode = MODE_WORD

        self._frame_lock = Lock()
        self._state_lock = Lock()
        self._frame_queue = deque(maxlen=1)
        self._sequence = deque(maxlen=SEQUENCE_LENGTH)
        self._sentence = []         # legacy — kept for _predict_sentence return
        self._smoother = TemporalSmoother(window=SMOOTHING_WINDOW, min_votes=SMOOTHING_MIN_VOTES)
        self._threshold = DEFAULT_THRESHOLD
        self._show_landmarks = True
        self._session_version = 0
        self._last_model_ms = 0.0
        self._last_emitted_word = None   # dedup word_detected signal

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
            # Use model_complexity=1 to match training landmark extraction exactly
            self._holistic = self._assets.mp_holistic.Holistic(
                model_complexity=1,
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
                with self._state_lock:
                    show_landmarks = self._show_landmarks
                if show_landmarks:
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

    def set_show_landmarks(self, value: bool):
        """Toggle drawing of landmarks on the feed."""
        with self._state_lock:
            self._show_landmarks = value

    def set_mode(self, mode: str):
        """Switch between word-level (MODE_WORD) and fingerspelling (MODE_FINGER)."""
        if mode not in (MODE_WORD, MODE_FINGER):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'word' or 'finger'.")
        with self._state_lock:
            self._mode = mode
            # Reset buffers on mode switch to avoid stale context
            self._sequence.clear()
            self._sentence.clear()
            self._smoother.reset()
            self._last_emitted_word = None
        self.status_updated.emit(f"Mode: {'Word-level ASL' if mode == MODE_WORD else 'Fingerspelling'}")

    def stop(self, wait_ms: int = 3000) -> bool:
        """Request a clean stop and wait briefly for inference to finish."""
        self._running = False

        if not self.isRunning():
            return True

        self.requestInterruption()

        if not self.wait(wait_ms):
            message = "Inference thread did not stop within the timeout."
            print(f"[InferenceThread] WARNING: {message}")
            # Do not emit error_occurred here to avoid triggering recursive shutdown loops
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

        checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)
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
            d_model=int(checkpoint.get("d_model", 192)) if isinstance(checkpoint, dict) else 192,
            nhead=int(checkpoint.get("nhead", 6)) if isinstance(checkpoint, dict) else 6,
            num_layers=int(checkpoint.get("num_layers", 3)) if isinstance(checkpoint, dict) else 3,
            dim_ff=int(checkpoint.get("dim_ff", 384)) if isinstance(checkpoint, dict) else 384,
            dropout=float(checkpoint.get("dropout", 0.0)) if isinstance(checkpoint, dict) else 0.0,
            seq_length=SEQUENCE_LENGTH,
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device)
        self._model.eval()

        # Load fingerspelling model if available (Task 6)
        self._load_finger_model()

        self.status_updated.emit(f"Model loaded: {self._model_path.name}")
        print(f"[InferenceThread] PyTorch checkpoint loaded: {self._model_path.name}")

    def _load_finger_model(self):
        """Load fingerspelling classifier if the weights exist (Task 6)."""
        if not FINGER_MODEL_PATH.exists():
            print(f"[InferenceThread] Fingerspelling model not found at {FINGER_MODEL_PATH} — finger mode will be unavailable.")
            return

        try:
            from core.fingerspelling_model import FingerspellingClassifier
            ckpt = torch.load(FINGER_MODEL_PATH, map_location=self._device, weights_only=False)
            num_classes = ckpt.get("num_classes", 27)
            self._finger_model = FingerspellingClassifier(num_classes=num_classes)
            self._finger_model.load_state_dict(ckpt["model_state_dict"])
            self._finger_model.to(self._device)
            self._finger_model.eval()
            self._finger_actions = ckpt.get("classes", [chr(i) for i in range(65, 91)] + [" "])
            print(f"[InferenceThread] Fingerspelling model loaded: {num_classes} classes.")
        except Exception as exc:
            print(f"[InferenceThread] WARNING: Could not load fingerspelling model: {exc}")

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
            self._last_emitted_word = None

    def _predict_sentence(self, keypoints) -> str:
        """Route to word-level or fingerspelling prediction based on current mode."""
        with self._state_lock:
            current_mode = self._mode

        if current_mode == MODE_FINGER:
            return self._predict_fingerspelling(keypoints)
        return self._predict_word(keypoints)

    def _predict_word(self, keypoints) -> str:
        """Word-level ASL prediction using GestureTransformer."""
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
                    # Emit word_detected for CaptionBuffer / LLM trigger (Task 9)
                    if predicted_word != self._last_emitted_word:
                        self._last_emitted_word = predicted_word
                        self.word_detected.emit(predicted_word)

            if len(self._sentence) > 5:
                self._sentence = self._sentence[-5:]

            return " ".join(self._sentence)

    def _predict_fingerspelling(self, keypoints) -> str:
        """Single-frame fingerspelling letter prediction (Task 6/7)."""
        if self._finger_model is None:
            return "[Fingerspell model not loaded]"

        # Only use right-hand landmarks (195:258 = rh, 63 floats)
        rh_keypoints = keypoints[195:258].astype(np.float32)
        input_tensor = torch.from_numpy(rh_keypoints).unsqueeze(0).to(self._device)

        model_started_at = time.perf_counter()
        with torch.inference_mode():
            logits = self._finger_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self._last_model_ms = (time.perf_counter() - model_started_at) * 1000.0

        predicted_index = int(np.argmax(probs))
        confidence = float(probs[predicted_index])

        with self._state_lock:
            threshold = self._threshold

        if confidence < threshold:
            return " ".join(self._sentence)

        letter = self._finger_actions[predicted_index]

        with self._state_lock:
            if not self._sentence or self._sentence[-1] != letter:
                self._sentence.append(letter)
                if len(self._sentence) > 30:   # fingerspelling can be long
                    self._sentence = self._sentence[-30:]
                if letter != self._last_emitted_word:
                    self._last_emitted_word = letter
                    self.word_detected.emit(letter)
            return "".join(self._sentence)
