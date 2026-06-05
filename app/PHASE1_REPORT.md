# Gestura - Phase 1 Implementation Report

Date: April 24, 2026
Project: Gestura APP

## 1. Executive Summary

Phase 1 focused on stabilizing and production-hardening the real-time sign-language subtitle pipeline.

The main outcomes were:
- Decoupled virtual camera output from inference/UI to remove frame pipeline stalls.
- Added startup validation to prevent silent runtime crashes.
- Added camera index selection in UI (instead of hardcoded camera 0).
- Cleaned runtime module boundaries (inference assets separated from data-collection script).
- Resolved major dependency compatibility conflicts (TensorFlow + MediaPipe + Protobuf + OpenCV + NumPy).

Result: The app now runs with a significantly more stable, debuggable, and user-controllable architecture.

---

## 2. Scope Completed in Phase 1

### 2.1 CaptureThread
- Capture moved/kept as an independent worker thread.
- Emits frames and FPS separately.
- Includes stop timeouts and camera failure signaling.

### 2.2 InferenceThread
- Performs MediaPipe + model inference off the UI thread.
- Uses bounded queue behavior to keep freshest frames.
- Emits status and error signals to UI.
- Clean stop and interruption handling added.

### 2.3 TemporalSmoother
- Integrated for stable word output and reduced subtitle flicker.

### 2.4 Main UI
- Receives annotated results for display.
- Shows status, FPS, and threshold controls.
- Handles thread errors and start/stop lifecycle robustly.

### 2.5 VirtualCamThread (Major Fix)
- Implemented dedicated virtual camera thread.
- Virtual camera send is isolated from inference/UI.
- Uses bounded queue (`maxlen=2`) + FPS pacing to avoid backlog.

### 2.6 Startup Checks
- Added model file check (`model2.keras`).
- Added camera availability check for selected index.
- Added optional pyvirtualcam check with warning-level handling.

### 2.7 Camera Selector
- Added camera index selector to UI using a spinbox.
- Recreates and rewires `CaptureThread` on index change.
- Disables selector while running to avoid mid-stream state corruption.

---

## 3. Key Difficulties Faced and Their Solutions

## Difficulty A: Virtual camera output caused slowdown/lag

### Problem
`pyvirtualcam.send()` is blocking. When called in the UI result path, the entire pipeline (especially inference responsiveness) slowed down waiting for meeting apps to consume frames.

### Root Cause
The virtual cam write path was coupled to frame display/inference result flow.

### Solution
- Introduced `VirtualCamThread` with its own queue.
- Routed same inference output to:
  - UI rendering path
  - Virtual cam thread path
- Kept virtual cam on independent cadence (around 20 FPS).

### Outcome
Inference no longer stalls because of meeting app pull-rate behavior.

---

## Difficulty B: MediaPipe import/API mismatch (`mp.solutions` missing)

### Problem
Runtime crashed with:
- `AttributeError: module 'mediapipe' has no attribute 'solutions'`

### Root Cause
Installed MediaPipe build/version exposed tasks API behavior inconsistent with the legacy `solutions` path required by the current code.

### Solution
Pinned MediaPipe to a version aligned with current code usage:
- `mediapipe==0.10.14`

### Outcome
`mp.solutions.holistic` became available again for runtime inference path.

---

## Difficulty C: TensorFlow-Protobuf-MediaPipe dependency conflicts

### Problem
Conflicting constraints caused runtime import errors and resolver conflicts, including protobuf version incompatibilities.

### Root Cause
- Newer TensorFlow (2.21.0) expected newer protobuf range.
- MediaPipe setup and runtime path expected protobuf range that did not match that stack.

### Solution
Standardized a known compatible stack:
- `tensorflow==2.15.0`
- `keras==2.15.0`
- `mediapipe==0.10.14`
- `protobuf==4.25.9`
- `numpy==1.26.4`
- `opencv-contrib-python==4.8.1.78`

### Outcome
Imports and app startup stabilized.

---

## Difficulty D: Runtime depended on ASL.py (mixed concerns)

### Problem
`ASL.py` mixed data collection/training/runtime concerns. Runtime imports from it created confusion and unnecessary coupling.

### Root Cause
Inference helpers and model assets were defined in a script originally used for training/data tasks.

### Solution
Created dedicated runtime module:
- `core/inference_assets.py`

Updated inference runtime to import from that module instead of `ASL.py`.

### Outcome
Cleaner separation between training/data scripts and runtime app path.

---

## Difficulty E: Silent failures and poor startup feedback

### Problem
App could fail deep in threads (camera/model issues) without clear user-facing diagnostics.

### Root Cause
Missing preflight validation and centralized startup checks.

### Solution
Added `core/startup_checks.py` and launch-time gating in UI.

### Outcome
Users receive clear actionable errors before threads start.

---

## Difficulty F: Hardcoded camera index

### Problem
Camera index fixed at 0 with no UI control.

### Solution
Added camera selector in UI and safe thread re-init workflow.

### Outcome
User can switch camera source without editing code.

---

## 4. Architecture After Phase 1

Current high-level runtime flow:

1. `CaptureThread` acquires frames.
2. `InferenceThread` performs MediaPipe + model prediction.
3. Inference emits result signal.
4. UI renders subtitle-baked preview.
5. Same result is routed to `VirtualCamThread` queue.
6. `VirtualCamThread` pushes to virtual camera independently.

This architecture ensures blocking virtual-cam operations do not slow inference.

---

## 5. Files Added / Updated

### Added
- `threads/virtual_cam_thread.py`
- `core/startup_checks.py`
- `core/inference_assets.py`
- `PHASE1_REPORT.md`

### Updated
- `ui/main_window.py`
- `threads/inference_thread.py`
- `requirements.txt`

---

## 6. Dependency Baseline (Pinned)

Phase 1 stable set:
- `tensorflow==2.15.0`
- `keras==2.15.0`
- `mediapipe==0.10.14`
- `protobuf==4.25.9`
- `numpy==1.26.4`
- `opencv-contrib-python==4.8.1.78`
- `PyQt6`
- `pyvirtualcam`
- `matplotlib`
- `scikit-learn`

---

## 7. Validation Performed

Validation checks performed during implementation:
- Static error scan on modified files.
- Import smoke tests for:
  - `threads.virtual_cam_thread`
  - `core.startup_checks`
  - `ui.main_window`
- Runtime dependency checks for MediaPipe, TensorFlow, NumPy, OpenCV compatibility.

---

## 8. Phase 1 Completion Status

Phase 1 checklist status:
- CaptureThread: Complete
- InferenceThread: Complete
- TemporalSmoother: Complete
- Main UI: Complete
- VirtualCamThread: Complete
- Error handling: Complete
- Camera selector: Complete

Overall status: Phase 1 complete.

---

## 9. Recommended Next Steps (Phase 2 Preview)

1. Performance optimization:
- Avoid duplicate subtitle drawing for preview and virtual cam by sharing annotated buffers safely.

2. UX improvements:
- Add explicit virtual-cam device name display and reconnect button.

3. Observability:
- Add lightweight runtime metrics panel (inference latency, queue depth, dropped frames).

4. Packaging:
- Freeze and export a reproducible environment lock file for deployment consistency.
