# Run 3 — GestureTransformer WLASL-300 (Anti-Overfitting)

## Why Run 3 Exists

Run 2 achieved 39.6% val accuracy but had a **38.6-point train-val gap** (76.9% train vs 39.6% val). Classic overfitting — the model memorized the ~8 training samples per word instead of learning actual gesture patterns. Run 3 attacks that gap directly with 5 simultaneous fixes.

---

## 5 Fixes Applied vs Run 2

### Fix 1 — Stronger Augmentation (`dataset.py`)
With only ~8 samples/word, augmentation is literally the only source of extra data.

| Augmentation | Run 2 | Run 3 |
|---|---|---|
| Gaussian noise σ | 0.005 | **0.03** (6×) |
| Frame dropout | 5% | **25%** |
| Speed warp range | 0.85–1.15× | **0.70–1.30×** |
| Augment probability | 0.3–0.5 | **0.5–0.7** |
| Hand scaling | ❌ | **0.85–1.15×** (new) |
| Spatial jitter | ❌ | **±0.05 x,y** (new) |

### Fix 2 — Mixup Training α=0.2 (`train.py`)
Blends two training samples (60% "accident" + 40% "about") into synthetic data. Label becomes soft `[0.6, 0.4, 0...]` instead of hard `[1, 0, 0...]`. Forces smooth decision boundaries.

### Fix 3 — Label Smoothing 0.05 → 0.2 (`train.py`)
Prevents the model from assigning 99% probability to memorized training examples.

### Fix 4 — Early Stop on val_loss, patience 50 → 40 (`train.py`)
Val accuracy is noisy with small datasets (one sample flip = 0.15% change). Val loss is a smoother signal.

### Fix 5 — Model Capacity Reduced 3.5M → ~1.8M params (`model.py`)

| Parameter | Run 2 | Run 3 |
|---|---|---|
| d_model | 256 | **192** |
| nhead | 8 | **6** |
| num_layers | 4 | **3** |
| dim_feedforward | 512 | **384** |
| dropout | 0.3 | **0.4** |
| Total params | ~3.5M | **~1.8M** |

---

## Expected Outcomes

| Metric | Run 2 | Run 3 Target |
|---|---|---|
| Train accuracy | 77.7% | ~55–60% |
| Val accuracy | 39.6% | **48–55%** |
| Test top-1 | 30.4% | **~45–52%** |
| Train-val gap | 38.6 pts | **~8–12 pts** |

> Note: Train accuracy *dropping* is a success signal — it means the model memorizes less.

---

## Pre-requisites

### Step 1 — Install CUDA PyTorch (critical for speed)

The `.venv` currently has CPU-only PyTorch (`2.12.0+cpu`). Training on CPU will take 10–20× longer.

```powershell
# In the Gestura v2 root
.\.venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True  |  NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

### Step 2 — Ensure landmarks_300 data is present

The training script expects `landmarks_300/` at:
```
Gestura v2/
└── landmarks_300/
    ├── train/   (300 word dirs with .npy files)
    ├── val/
    ├── test/
    ├── manifest.json
    └── label_map.json
```

If you need to re-extract landmarks from the WLASL_300 videos:
```powershell
# Put WLASL_300/ videos in: Gestura v2\WLASL_300\
.\.venv\Scripts\python.exe training\Model\extract_landmarks.py --workers 4
```

---

## Launching Run 3

### Option A — PowerShell Launch Script (recommended)
```powershell
cd "c:\Users\Saksham jain\Desktop\Gestura\Gestura v2\training"
.\run3_launch.ps1
```

**Or with a custom landmarks path:**
```powershell
.\run3_launch.ps1 -LandmarksDir "C:\path\to\landmarks_300"
```

**Quick test (3 epochs, 50 samples, verify pipeline works):**
```powershell
.\run3_launch.ps1 -QuickTest
```

**Resume from checkpoint:**
```powershell
.\run3_launch.ps1 -Resume "Model\gesture_model_300.pt"
```

### Option B — Direct Python
```powershell
cd "c:\Users\Saksham jain\Desktop\Gestura\Gestura v2\training\Model"
..\..\..\.venv\Scripts\python.exe train.py `
    --epochs 500 `
    --batch-size 32 `
    --num-workers 4 `
    --d-model 192 `
    --nhead 6 `
    --num-layers 3 `
    --dim-ff 384 `
    --dropout 0.4 `
    --mixup-alpha 0.2 `
    --patience 40 `
    --landmarks-dir "..\..\landmarks_300"
```

### Pre-flight Check
```powershell
.\.venv\Scripts\python.exe training\preflight_check.py
```

---

## Output Files

```
training/
├── Model/
│   ├── gesture_model_300.pt          ← Best checkpoint (saves every improvement on val_loss)
│   ├── gesture_model_300_inference.pt ← Final inference-ready model
│   └── checkpoint_epoch_N.pt         ← Periodic checkpoints (every 10 epochs)
└── training_logs/
    └── training_history.json         ← Epoch-by-epoch metrics
```

---

## After Training — Organize Results

```powershell
# Move results into a structured Run_3 directory
.\.venv\Scripts\python.exe training\Results\organize_run.py `
    --run-name Run_3 `
    --description "Anti-overfitting: mixup alpha=0.2, label_smooth=0.2, d=192, 1.8M params"
```

---

## Training Settings Summary

| Setting | Value |
|---|---|
| Optimizer | AdamW (weight_decay=5e-5) |
| Learning rate | 5e-4 → 1e-6 cosine annealing |
| Warmup | 10 epochs linear ramp |
| Batch size | 32 |
| Max epochs | 500 |
| Early stopping | patience=40, monitor=val_loss |
| Mixed precision | AMP (GPU only) |
| Gradient clipping | max_norm=1.0 |
| Mixup | α=0.2 |
| Label smoothing | 0.2 |
| Class imbalance | WeightedRandomSampler |
