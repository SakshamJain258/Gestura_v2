# Gestura v2

Modular Gestura Phase 1 app with PyQt6 UI, threaded capture/inference, TensorFlow/Keras support, and PyTorch gesture model assets.

## Setup

For CPU/default PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA PyTorch, install the CUDA wheel first, then install the remaining requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Run

```powershell
cd app
python app.py
```

## Structure

- `app/` contains the runnable Phase 1 app.
- `training/` contains training code, selected results, and trained model assets.
- `workflow/` contains workflow/reference assets.

Keep `.venv/` local. It is ignored by Git.
