# 06 — Camera Pipeline

## Overview

Getting a reliable, responsive webcam pipeline on Windows turned out to be one of the most significant engineering challenges of this project. This document records every bug encountered and the final design that solved them.

---

## Windows Camera Backends

OpenCV on Windows supports three primary backends for `cv2.VideoCapture`:

| Backend | Flag | Notes |
|---|---|---|
| **MSMF** (Media Foundation) | `cv2.CAP_MSMF` | Windows default. Fast in theory, but buggy with Python OpenCV on many systems. |
| **DirectShow** | `cv2.CAP_DSHOW` | Legacy Windows API. More stable for Python scripts. Slower to open, but reliable. |
| **Default** | *(none)* | OpenCV auto-selects; on Windows this is usually MSMF first. |

---

## Bugs Encountered

### Bug 1: Setting `CAP_PROP_FPS` deadlocks the camera driver

**Symptom:** The app would freeze completely for 30–180+ seconds after clicking Start, or the camera would stop responding across all apps (Zoom, Teams, Windows Camera) until the Python process was force-killed.

**Root Cause:** Calling `cap.set(cv2.CAP_PROP_FPS, 25)` on a Windows webcam driver triggers a format negotiation sequence inside the DirectShow/MSMF filter graph. Many webcam drivers do not handle this negotiation correctly at the Python/OpenCV layer and enter a deadlocked state. The deadlock prevents `cap.release()` from completing, leaving the device handle open indefinitely.

**Diagnostic evidence:**
```
# Original diagnostic test results (with FPS setting):
Setting resolution to 640x480 took: 10813.3043 seconds  ← driver deadlock
```

**Fix:**
```python
# REMOVED — causes Windows driver deadlock
# cap.set(cv2.CAP_PROP_FPS, 25)

# Only set resolution (safe on all tested drivers):
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

### Bug 2: Default backend opens but returns no frames (silent failure)

**Symptom:** The camera feed panel stayed completely blank after clicking Start. No subtitle appeared. No error dialog. The app appeared to be running normally.

**Root Cause:** On Windows, `cv2.VideoCapture(0)` with the MSMF backend can report `isOpened() == True` but then return `ret=False` on every `cap.read()`. The camera appears acquired but is actually in a broken state. OpenCV 4.8 has known issues with MSMF on certain webcam driver combinations.

**Why no error dialog?** The original code only checked `isOpened()`, which returned `True`. It never attempted a frame read before entering the main loop.

**Diagnostic evidence:**
```
--- Testing with backend: Default ---
Open attempt took: 100.8291 seconds
Camera opened successfully with Default!     ← isOpened() = True
Frame 1 failed to read in 0.0130 seconds.   ← but read() fails
```

**Fix:** Added a validation frame read before entering the capture loop:
```python
cap = cv2.VideoCapture(self.camera_index)

# Validate: try to read an actual frame
working = False
if cap.isOpened():
    ret, _ = cap.read()
    if ret:
        working = True
    else:
        cap.release()  # abandon broken backend

if not working:
    # Fall back to DirectShow
    print("[CaptureThread] Default backend failed. Falling back to CAP_DSHOW...")
    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
```

---

### Bug 3: Recursive shutdown loop causes repeated timeout warnings

**Symptom:**
```
[CaptureThread] WARNING: Camera thread did not stop within the timeout.
[CaptureThread] WARNING: Camera thread did not stop within the timeout.
[CaptureThread] WARNING: Camera thread did not stop within the timeout.
[CaptureThread] WARNING: Camera thread did not stop within the timeout.
[CaptureThread] WARNING: Camera thread did not stop within the timeout.
[CaptureThread] WARNING: Camera thread did not stop within the timeout.
[CaptureThread] Camera released.
```

**Root Cause:**
1. `CaptureThread.stop()` timed out (camera driver slow to release)
2. It emitted `error_occurred(message)`
3. `MainWindow._on_runtime_error()` handled this by calling `_on_stop()`
4. `_on_stop()` checked `if not self._is_running: return` — but `_is_running` was still `True` because it was set to `False` at the *end* of `_on_stop()`, not the beginning
5. So `_on_stop()` ran again → called `stop()` again → timed out → emitted `error_occurred` → loop

**Fix 1:** Remove `error_occurred.emit()` from thread `stop()` methods — a timeout during cleanup is not a runtime error:
```python
def stop(self, wait_ms=2000):
    self._running = False
    self.requestInterruption()
    if not self.wait(wait_ms):
        print("[CaptureThread] WARNING: Camera thread did not stop within the timeout.")
        # Do not emit error_occurred — avoids recursive shutdown loop
        return False
    return True
```

**Fix 2:** Set `self._is_running = False` at the very start of `_on_stop()` to act as a re-entry guard:
```python
def _on_stop(self):
    if not self._is_running:
        return
    self._is_running = False  # ← guard here, not at the end
    ...
```

---

### Bug 4: Orphaned Python processes lock the camera across reboots

**Symptom:** After stopping the app, the webcam is unavailable in Teams, Windows Camera, or in the next run of Gestura until a restart.

**Root Cause:** When `cap.release()` hangs (due to bugs 1 or 2), the Python process stays alive as a zombie in the background. The Windows camera driver sees the device handle as still open, so all other apps are blocked.

**Fix:** Resolved by fixing bugs 1 and 2 (proper release now happens cleanly). Additionally, the app now validates frame reading during startup rather than discovering the broken state inside the main loop.

**Emergency recovery** (if camera still locked after a crash):
```powershell
# Find orphaned python processes:
Get-Process -Name python | Select-Object Id, Path
# Kill them:
Stop-Process -Id <ID> -Force
```

---

## Final Camera Initialization Code

```python
def run(self):
    self._running = True
    
    # Try default backend first, validate with frame read
    cap = cv2.VideoCapture(self.camera_index)
    working = False
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            working = True
        else:
            cap.release()
    
    if not working:
        print("[CaptureThread] Default backend failed. Falling back to CAP_DSHOW...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        self.error_occurred.emit(f"Could not open camera {self.camera_index}.")
        self._running = False
        return

    # Second validation
    ret, _ = cap.read()
    if not ret:
        ret, _ = cap.read()
        if not ret:
            cap.release()
            self.error_occurred.emit(f"Camera {self.camera_index} opened but returned no frames.")
            self._running = False
            return

    # Safe to configure resolution (do NOT set FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ...
```

---

## Inference-Side Resolution

The camera captures at `640×480`. The inference thread additionally downscales to `640×360` (16:9) before passing to MediaPipe:

```python
frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
```

This keeps MediaPipe running at consistent speed regardless of the camera's native resolution.

---

## Landmark Overlay

A "Show Landmarks Overlay" checkbox was added to the UI. When enabled, MediaPipe's skeletal drawing is rendered on the feed (pose skeleton + left/right hand meshes). This is critical for debugging: if the overlay is not tracking hands, MediaPipe is not detecting them — which means the model will receive all-zero hand inputs and cannot predict correctly.

To debug tracking failures:
1. Enable the overlay
2. Ensure your upper body and both hands are fully visible in the frame
3. Improve lighting (avoid backlighting, ensure even illumination on hands)
4. Sit 1–2 metres from the camera
