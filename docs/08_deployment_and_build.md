# 08 — Deployment and Build

> **Current status:** Gestura v2 includes a production-ready PyInstaller build system that packages the desktop application into a standalone Windows directory. Users can run the app without installing Python by launching the bundled `Gestura.exe`.

---

## Bundled Windows Distribution (Recommended)

Gestura v2 uses a **one-folder** PyInstaller distribution. This mode is chosen because:
1. PyTorch (~500MB) and MediaPipe contain large binary DLL trees that launch instantly in one-folder mode (one-file mode would require extracting 500MB+ of DLLs to a temporary directory on every single launch, causing a 10-15 second delay).
2. Standard data files for MediaPipe, OpenCV, and label maps are collected and resolved correctly using a frozen-exe-aware path resolver.

---

## Running the App from Source (Python Script)

Alternatively, the application can be run directly using Python:

```powershell
# From the repo root:
cd app
python app.py
```

This requires:
- Python 3.11 installed
- All dependencies installed in the virtual environment
- Model weights at the expected paths (see README for details)

---

## Dependency Pinned Stack

The following version combination was validated as stable on Windows 11. Changing any of these — especially **MediaPipe**, **protobuf**, or **numpy** — may break the app.

| Package | Version | Why pinned |
|---|---|---|
| `mediapipe` | `0.10.14` | Uses legacy `mp.solutions` API; newer versions removed it |
| `opencv-contrib-python` | `4.8.1.78` | Newer versions change MSMF backend behavior |
| `protobuf` | `4.25.3` | Must stay within MediaPipe's accepted range |
| `numpy` | `1.26.4` | Required by mediapipe and OpenCV |
| `torch` | latest 2.x | Or CUDA build (see below) |
| `PyQt6` | latest | |
| `pyvirtualcam` | latest | Optional — only needed for virtual camera |
| `google-generativeai` | latest | Or `google-genai` (new SDK) — optional |

### Installing dependencies

**CPU only (simpler, works on any machine):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**CUDA (recommended if you have an NVIDIA GPU):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Virtual Camera (Optional)

The virtual camera feature pipes the annotated video feed into Zoom, Teams, or Google Meet.

### Prerequisites

1. Install **OBS Studio**: [obsproject.com](https://obsproject.com/)
2. In OBS, go to **Tools → Virtual Camera → Start Virtual Camera** at least once to register the driver
3. The `pyvirtualcam` Python package automatically routes to this driver

**Without a virtual camera driver:** the app runs normally — the "Virtual Camera" checkbox simply has no effect.

---

## Gemini API Key

The Gemini key is stored locally at `~/.gestura/config.json` and is never in the repository. Users configure it through the app's **⚙ Key** dialog.

---

## Building the Executable from Source

To package Gestura yourself from source, follow these steps:

### 1. Build Prerequisite Checklist
Before running the build, ensure:
- Your virtual environment is configured at `.venv` in the repository root and dependencies are fully installed.
- The required model weights exist in their designated folders (these are required during the packaging process to be bundled in the distribution):
  - `training/Results/Run_3/gesture_model_300_inference.pt`
  - `training/Results/Fingerspelling/fingerspelling_model.pt`
  - `landmarks_300/label_map.json`

### 2. Run the Build Script
Run the automated build script from the repository root:
```powershell
cd build
.\build.ps1
```

This script:
1. Verifies the virtual environment and required models exist.
2. Auto-installs PyInstaller into the virtual environment if missing.
3. Cleans previous build and work directories (`dist/` and `build_temp/`).
4. Runs PyInstaller with `gestura.spec`.
5. Outputs a self-contained folder under `build/dist/Gestura/` containing `Gestura.exe`.

---

## Technical Details of the Build

The PyInstaller configuration in [`gestura.spec`](file:///c:/Users/Saksham%20jain/Desktop/Gestura/Gestura%20v2/build/gestura.spec) handles the following complex requirements:

1. **MediaPipe Native Binary Packaging**: MediaPipe bundles `.tflite` model files and protocol buffers that fail to load dynamically in frozen apps. The spec file utilizes PyInstaller's `collect_data_files("mediapipe")` to bundle them in the output.
2. **OpenCV Support**: Includes `collect_data_files("cv2")` to ensure haarcascades and OpenCV data files are copied.
3. **No UPX Compression**: UPX is disabled (`upx=False`) because UPX compression is known to corrupt PyTorch DLLs (`_C.pyd`, etc.) on Windows.
4. **Frozen Path Resolution**: The application code (`inference_assets.py`, `inference_thread.py`, and `startup_checks.py`) uses a helper function `_get_base_dir()` to dynamically resolve resource paths. When `getattr(sys, 'frozen', False)` is true, it routes to `sys._MEIPASS` (the directory where resources are unpacked at runtime) rather than `__file__`.

---

## Troubleshooting Common Launch Issues

| Symptom | Cause | Fix |
|---|---|---|
| `Model file not found` error | `.pt` file missing | Download from Releases page (see README) |
| Camera feed blank, no error | Default OpenCV backend silent failure | App auto-falls back to CAP_DSHOW. If still blank, check if another app (Teams, Zoom) is holding the camera. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Wrong MediaPipe version | `pip install mediapipe==0.10.14` |
| `protobuf version mismatch` | Version conflict | `pip install protobuf==4.25.3` |
| `ImportError: PyQt6` | PyQt6 not installed | `pip install PyQt6` |
| Gemini toggle greyed out | No API key configured | Click **⚙ Key** in the app |
| Camera still locked after crash | Orphaned Python process holding device handle | `Stop-Process -Name python -Force` in PowerShell |
