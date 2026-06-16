# 05 — App Architecture

## Overview

The Gestura v2 desktop app is built in PyQt6 and runs four background worker threads in parallel. The design principle is strict separation of concerns: each thread has one job and communicates only through Qt signals.

---

## File Map

```
app/
├── app.py                       # Entry point
├── core/
│   ├── inference_assets.py      # ML inference utilities (shared across app)
│   ├── caption_buffer.py        # Rolling word buffer + pause detection timer
│   ├── smoother.py              # TemporalSmoother (voting-based anti-flicker)
│   ├── mode_controller.py       # Word ↔ fingerspelling mode management
│   ├── pause_detector.py        # Detects signing pause → triggers Gemini
│   ├── llm_client.py            # GeminiClient wrapper
│   ├── fingerspelling_model.py  # FingerspellingClassifier MLP
│   └── startup_checks.py        # Pre-launch validation
├── threads/
│   ├── capture_thread.py        # Webcam frame acquisition
│   ├── inference_thread.py      # MediaPipe + model inference
│   └── virtual_cam_thread.py    # pyvirtualcam output
└── ui/
    ├── main_window.py           # Main PyQt6 window + signal wiring
    └── api_key_dialog.py        # Gemini API key dialog
```

---

## Threading Model

```
┌────────────────────────────────────────────────────────────┐
│ Main Thread (Qt Event Loop)                                 │
│   MainWindow  ← all UI updates (thread-safe via Qt signals) │
└────────────────────────────────────────────────────────────┘
        ↑ result_ready, word_detected, fps_updated, errors
        │
        │ frame_ready (raw frames)
        ↓
┌───────────────────┐      ┌───────────────────────────┐
│  CaptureThread    │ ───► │  InferenceThread           │
│  cv2.VideoCapture │      │  MediaPipe + Transformer   │
└───────────────────┘      └────────────┬──────────────┘
                                        │ push_frame(annotated)
                                        ▼
                           ┌───────────────────────────┐
                           │  VirtualCamThread          │
                           │  pyvirtualcam.send()       │
                           └───────────────────────────┘
```

---

## Signal Flow (Complete)

```
CaptureThread.frame_ready
    → InferenceThread.receive_frame()

InferenceThread.result_ready(frame, sentence)
    → MainWindow._on_result()              — draw subtitle, display frame
    → MainWindow._on_result_to_vcam()      — route to VirtualCamThread

InferenceThread.word_detected(word)
    → CaptionBuffer.append(word)
        → PauseDetector (2.5s timer resets on each word)
            → (on timeout) GeminiClient.refine_caption(words)
                → GeminiWorker (background QThread)
                    → MainWindow._on_llm_caption(text)

CaptureThread.error_occurred / InferenceThread.error_occurred
    → MainWindow._on_runtime_error() → _on_stop() + error dialog
```

---

## CaptureThread

**Job:** Acquire raw frames from the webcam.

**Design:**
- Does NOT run MediaPipe or model — just reads and emits frames
- Backend validation: reads a test frame before entering the loop; falls back to `CAP_DSHOW` if default backend returns no frames
- Does NOT call `cap.set(CAP_PROP_FPS, ...)` — this deadlocks Windows drivers
- Sets `640×480` resolution only
- Emits `fps_updated` every second

---

## InferenceThread

**Job:** Run MediaPipe and the PyTorch model.

**Design:**
- Internal `deque(maxlen=1)` — always processes the freshest frame; old frames are discarded if inference is slow
- Rolling `deque(maxlen=60)` — maintains the last 60 landmark vectors (the Transformer's input window)
- `MIN_SEQUENCE_FOR_PREDICTION = 20` — predictions only start after 20 frames are accumulated
- Frame padding: sequences shorter than 60 frames are padded by repeating the last frame

**Prediction path:**

```
frame → MediaPipe → keypoints (258,) → append to rolling deque
if len(deque) >= 20:
    snapshot = last 60 frames
    pad if needed
    → GestureTransformer → softmax → argmax + confidence
    → TemporalSmoother.update(index, confidence, threshold)
    → if stable_prediction: emit word_detected
```

---

## TemporalSmoother

**Problem:** At 25+ predictions/second, the subtitle would flicker on every frame.

**Solution:** Majority voting over a sliding window.

```python
class TemporalSmoother:
    def __init__(self, window=6, min_votes=3):
        self._history = deque(maxlen=window)
    
    def update(self, pred_index, confidence, threshold):
        self._history.append(pred_index if confidence >= threshold else -1)
        most_common, count = Counter(self._history).most_common(1)[0]
        if most_common != -1 and count >= self.min_votes:
            return most_common
        return None
```

A word only becomes a subtitle if it was the dominant prediction in ≥ `min_votes` of the last `window` frames AND its confidence passed the threshold slider.

---

## CaptionBuffer

**Job:** Maintain the running sentence (rolling last-5 words) and detect signing pauses.

- Caps at `max_words=5`
- Runs a 2.5-second `QTimer` — resets on each new word; fires when signing stops → triggers Gemini
- Thread-safe via Qt signals

---

## ModeController

Manages switching between word-level ASL (`MODE_WORD`) and fingerspelling (`MODE_FINGER`).

On mode switch:
1. Sets the mode flag on `InferenceThread`
2. Resets the rolling sequence, sentence, and smoother
3. Calls `_on_clear()` in the UI — prevents stale context from one mode leaking into the other

---

## VirtualCamThread

**Job:** Send annotated frames to `pyvirtualcam`.

**Design:**
- `pyvirtualcam.Camera.send()` can block for tens of milliseconds waiting for the meeting app — this thread isolates that from inference
- Uses `deque(maxlen=2)` — drops old frames if pyvirtualcam is slow
- Paces output at 20 FPS using wall-clock timing
- Lazy init — does not start until the first annotated frame arrives

---

## Startup Validation

`core/startup_checks.py` validates before any thread starts:

1. Model `.pt` file exists at the expected path
2. Camera index is ≥ 0
3. `pyvirtualcam` is importable (warning only if missing — does not block)

Failures show an error dialog and prevent the inference threads from starting.

---

## Re-entry Guard in _on_stop

A recursion bug existed where a camera timeout → error → `_on_stop()` → `stop()` → timeout → error → `_on_stop()` infinite loop occurred.

Fixed by setting `_is_running = False` at the very top of `_on_stop()`:

```python
def _on_stop(self):
    if not self._is_running:
        return          # re-entry guard
    self._is_running = False
    ...
```

And by not emitting `error_occurred` from thread `stop()` on timeout — a stop timeout is not a runtime error.
