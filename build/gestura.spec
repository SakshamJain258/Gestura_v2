# -*- mode: python ; coding: utf-8 -*-
"""
gestura.spec — PyInstaller spec for Gestura desktop app.

Build command (from the build/ directory):
    pyinstaller gestura.spec

Or use build.ps1 for a full clean + validation build.

Output: build/dist/Gestura/Gestura.exe  (one-FOLDER bundle)

One-folder is used instead of one-file because:
  - PyTorch and MediaPipe have large native DLL trees that UPX cannot compress well
  - One-file requires extracting all DLLs to a temp dir on every launch (~5-15 sec cold start)
  - One-folder launches instantly after the first run
"""

import sys
import os
import importlib.util
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── Resolve project paths ─────────────────────────────────────────────────────
SPEC_DIR     = Path(SPECPATH)              # build/
PROJECT_ROOT = SPEC_DIR.parent             # Gestura v2/
APP_DIR      = PROJECT_ROOT / "app"        # Gestura v2/app/

# ── Collect MediaPipe data files ──────────────────────────────────────────────
# MediaPipe ships .tflite model files and proto descriptors that must be
# bundled — without them, Holistic fails silently at runtime.
mediapipe_datas = collect_data_files("mediapipe")

# ── Collect cv2 data files ─────────────────────────────────────────────────────
cv2_datas = collect_data_files("cv2")

# ── App data files ─────────────────────────────────────────────────────────────
app_datas = [
    # Label map (300 ASL words)
    (str(PROJECT_ROOT / "landmarks_300" / "label_map.json"),
     "landmarks_300"),

    # GestureTransformer weights (Run 3)
    (str(PROJECT_ROOT / "training" / "Results" / "Run_3" / "gesture_model_300_inference.pt"),
     "training/Results/Run_3"),

    # Fingerspelling model weights
    (str(PROJECT_ROOT / "training" / "Results" / "Fingerspelling" / "fingerspelling_model.pt"),
     "training/Results/Fingerspelling"),
]

all_datas = mediapipe_datas + cv2_datas + app_datas

# ── Analysis ───────────────────────────────────────────────────────────────────
a = Analysis(
    [str(APP_DIR / "app.py")],
    pathex=[str(APP_DIR)],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        # ── PyQt6 ──────────────────────────────────────────────────────────
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtNetwork",
        # ── PyTorch ────────────────────────────────────────────────────────
        "torch",
        "torch.nn",
        "torch.nn.modules.transformer",
        "torch.backends",
        "torch.backends.cudnn",
        "torch.jit",
        # ── MediaPipe ──────────────────────────────────────────────────────
        "mediapipe",
        "mediapipe.python",
        "mediapipe.python.solutions",
        "mediapipe.python.solutions.holistic",
        "mediapipe.python.solutions.drawing_utils",
        "mediapipe.python.solutions.drawing_styles",
        # ── OpenCV ─────────────────────────────────────────────────────────
        "cv2",
        # ── NumPy ──────────────────────────────────────────────────────────
        "numpy",
        "numpy.core",
        # ── Gemini SDK (both old and new — user may have either) ───────────
        "google",
        "google.generativeai",
        "google.ai",
        "google.ai.generativelanguage",
        "google.genai",
        # ── pyvirtualcam (optional — soft import in app) ───────────────────
        "pyvirtualcam",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy training-only deps to keep bundle lean
        "tensorflow",
        "keras",
        "tensorboard",
        "sklearn",
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ── EXE (launcher stub for one-folder mode) ────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],                        # ← empty: binaries go into COLLECT, not the exe stub
    exclude_binaries=True,     # ← one-folder mode
    name="Gestura",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX disabled — breaks PyTorch DLLs on Windows
    console=False,             # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(SPEC_DIR / "gestura.ico"),
)

# ── COLLECT (assembles all files into dist/Gestura/) ──────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Gestura",
)
