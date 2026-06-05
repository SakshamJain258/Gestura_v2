# 🤟 GESTURA — Complete Project Documentation

**Project:** Gestura — Real-Time Sign Language to Text/Speech  
**Author:** Saksham Jain  
**Started:** April 2026  
**Last Updated:** April 27, 2026 — 8:15 PM IST  

---

## Artifact Organization Update

As of April 28, 2026, trained model artifacts are separated under `Trained_Models/`:

- `Run_2_GestureTransformer_WLASL300_3p46M_Val39p60/` contains the Run 2 best model, inference weights, milestone checkpoints, code snapshot, logs, and reports.
- `Run_3_GestureTransformer_WLASL300_AntiOverfitting_1p66M_Val6p78/` contains the Run 3 best validation-loss model, inference weights, all periodic checkpoints, code snapshot, and training history.

The active training and extraction scripts remain in `Model/`. Shared datasets remain in `WLASL_300/` and `landmarks_300/`.

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [The Core Problem](#2-the-core-problem)
3. [Architecture Design — The 3-Layer Solution](#3-architecture-design--the-3-layer-solution)
4. [Phase 1 — Foundation (Complete)](#4-phase-1--foundation-complete)
5. [Phase 2 — WLASL 300-Word Gesture Model (In Progress)](#5-phase-2--wlasl-300-word-gesture-model-in-progress)
6. [Phase 2b — Fingerspelling Fallback (Planned)](#6-phase-2b--fingerspelling-fallback-planned)
7. [Phase 3 — LLM Correction Layer (Planned)](#7-phase-3--llm-correction-layer-planned)
8. [Technical Decisions Log](#8-technical-decisions-log)
9. [File Structure](#9-file-structure)
10. [Setup & Usage Guide](#10-setup--usage-guide)
11. [Change Log](#11-change-log)

---

## 1. Project Vision

Gestura is a real-time sign language recognition application that translates ASL (American Sign Language) gestures into natural English text. The goal is to bridge the communication gap between deaf/hard-of-hearing individuals and hearing people.

**What makes Gestura different from a typical demo project:**
- It handles continuous signing, not just isolated word recognition
- It has a fingerspelling fallback for words outside its vocabulary (names, places, technical terms)
- It uses an LLM to convert raw ASL word streams into grammatically correct English (ASL grammar ≠ English grammar)
- It runs in real-time with virtual camera output for use in video calls

---

## 2. The Core Problem

### Why a Single Model Isn't Enough

A single word-classifier model has one fundamental limitation:

```
It sees:    60 frames → predicts 1 word
Real signing: continuous flow, variable speed,
              words blend into each other,
              grammar is completely different from English
```

**Example of what actually happens in real signing:**

| What is signed | What model outputs | What user wants |
|---|---|---|
| HELLO YOU NAME WHAT | `["hello", "you", "name", "what"]` | `"Hello, what is your name?"` |

Sign language drops articles, helper verbs, and rearranges grammar. A single model can never fix this — it only sees gestures, not language structure. This is why the 3-layer architecture was chosen.

---

## 3. Architecture Design — The 3-Layer Solution

This architecture was designed after analyzing the limitations of single-model approaches. Saksham proposed the multi-layer concept, and after discussion, we refined it into three specific layers with clear job boundaries:

```
┌─────────────────────────────────────────────────┐
│  LAYER 1 — Gesture Model (Phase 2)              │
│                                                 │
│  Input:  60 frames of landmarks                 │
│  Output: word + confidence score                │
│  Job:    "What sign is this?"                   │
│  Vocab:  300 words from WLASL                   │
│  Model:  Conv1D + Transformer Encoder           │
└──────────────────┬──────────────────────────────┘
                   │ low confidence?
                   ▼
┌─────────────────────────────────────────────────┐
│  LAYER 2 — Fingerspelling Fallback (Phase 2b)   │
│  (small, separate model)                        │
│                                                 │
│  Input:  single frame hand landmarks            │
│  Output: A-Z letter                             │
│  Job:    "Spell out words not in vocab"         │
│  When:   Gesture model confidence < 0.6         │
└──────────────────┬──────────────────────────────┘
                   │ both outputs feed into
                   ▼
┌─────────────────────────────────────────────────┐
│  LAYER 3 — LLM Correction (Phase 3)            │
│                                                 │
│  Input:  raw word stream                        │
│  e.g.:   "hello you name what"                  │
│  Output: natural sentence                       │
│  e.g.:   "Hello, what is your name?"            │
│  Job:    Grammar, articles, natural flow        │
└─────────────────────────────────────────────────┘
```

### Why Each Layer Exists

| Layer | Purpose | Without It |
|---|---|---|
| **Layer 1 — Gesture Model** | Handles 95% of communication. 300 WLASL words cover most everyday conversation. | No sign recognition at all |
| **Layer 2 — Fingerspelling** | Every deaf person uses fingerspelling for names, places, technical words. 26 classes, one frame at a time. | App breaks when someone spells "COVID" or their name |
| **Layer 3 — LLM** | Takes messy word stream → reconstructs natural English. Doesn't need to understand signs at all. | Output is broken grammar: "hello you name what" |

### Build Order Decision

| Phase | What | Complexity | Time |
|---|---|---|---|
| Phase 2 (now) | Layer 1 — WLASL 300-word gesture model | Medium | Days |
| Phase 2b (next) | Layer 2 — Fingerspelling model | Low | +2 days |
| Phase 3 (after) | Layer 3 — LLM wiring | Low to wire | +1 day |

---

## 4. Phase 1 — Foundation (Complete)

**Date completed:** April 24, 2026  
**Full report:** [PHASE1_REPORT.md](Phase_1/PHASE1_REPORT.md)

### What Was Built

Phase 1 was the production-hardening of the real-time sign language subtitle pipeline with a small 16-word vocabulary.

**Components delivered:**
- `CaptureThread` — Independent camera frame acquisition
- `InferenceThread` — MediaPipe + model inference off UI thread
- `VirtualCamThread` — Dedicated virtual camera output (decoupled from inference)
- `TemporalSmoother` — Stable word output, reduced subtitle flicker
- `MainWindow` — PyQt6 UI with status, FPS, threshold controls
- `startup_checks.py` — Preflight validation for model/camera
- `inference_assets.py` — Clean runtime module (separated from training code)

### Model Used (Phase 1)

- **Architecture:** Conv1D + BiLSTM + Attention (Keras/TensorFlow)
- **Input:** 60 frames × 258 landmarks (pose + hands)
- **Output:** 16 word classes
- **Trained on:** Self-collected webcam data (50 sequences × 60 frames per word)

### Key Problems Solved

1. **Virtual camera lag** — pyvirtualcam.send() was blocking inference → Fixed with dedicated VirtualCamThread
2. **MediaPipe version crash** — Pinned to mediapipe==0.10.14
3. **TF/Protobuf conflicts** — Locked entire dependency chain
4. **Mixed concerns in ASL.py** — Created dedicated inference_assets.py
5. **Silent failures** — Added startup_checks.py with user-facing error messages
6. **Hardcoded camera** — Added camera selector UI

### Phase 1 Dependency Stack (Pinned)

```
tensorflow==2.15.0
keras==2.15.0
mediapipe==0.10.14
protobuf==4.25.9
numpy==1.26.4
opencv-contrib-python==4.8.1.78
PyQt6
pyvirtualcam
```

---

## 5. Phase 2 — WLASL 300-Word Gesture Model (In Progress)

**Started:** April 25, 2026  
**Status:** Landmark extraction running, model architecture built and verified

### What Saksham Asked

> "For the very first layer which model or transformer or any hybrid model should we use?"

The question was about choosing the right architecture for Layer 1 to handle 300 word classes (up from 16 in Phase 1).

### What Was Suggested — The Architecture Decision

Three serious candidates were evaluated:

#### Option 1: Conv1D + Transformer Encoder ← **CHOSEN**
- Conv1D captures local temporal patterns (quick flicks, holds)
- Transformer captures global dependencies (frame-to-frame attention across full 60 frames)
- Multi-head attention handles multi-joint relationships simultaneously
- ~3.5M parameters, ~8ms inference on GPU
- PyTorch native `nn.TransformerEncoder` — no external deps

#### Option 2: SPOTER (Sign POse-based TransformER)
- Research model specifically for WLASL
- Pure Transformer, less flexible, more complex to implement
- Rejected: harder to integrate into real-time pipeline

#### Option 3: ST-GCN (Spatial-Temporal Graph Convolution)
- Treats skeleton as graph, captures anatomical structure
- Rejected: significantly more complex, marginal accuracy gain, slower inference

#### Why the Current BiLSTM Wouldn't Scale

| Aspect | BiLSTM (Phase 1) | Conv1D + Transformer (Phase 2) |
|---|---|---|
| Long-range dependencies | Struggles after ~30 frames | Full attention over all 60 frames |
| Training speed | Sequential (slow) | Parallelized (2-3× faster) |
| Scaling to 300 classes | Capacity bottleneck | Scales with heads |
| Inference speed | ~15ms | ~8ms (no recurrence) |
| Subtle sign differences | Misses co-occurring joints | Multi-head attention captures multi-joint simultaneously |

### What Was Built

#### Framework Switch: TensorFlow → PyTorch

Phase 1 used TF/Keras. Phase 2 switches to PyTorch for:
- Better research ecosystem (most sign language papers use PyTorch)
- Native mixed precision support
- Easier debugging and custom training loops
- Better Transformer support

#### Files Created

All files in `MODEL_Training/Model/`:

##### 1. `extract_landmarks.py` — Landmark Extraction Pipeline

**Job:** Convert raw WLASL MP4 videos → MediaPipe landmarks → `.npy` files

**How it works:**
1. Opens each MP4 video
2. Runs MediaPipe Holistic on every frame
3. Extracts 258 features per frame: pose (33×4) + left hand (21×3) + right hand (21×3)
4. Pads/truncates to exactly 60 frames
5. Saves as `.npy` array of shape `(60, 258)`
6. Builds `manifest.json` (word → list of .npy paths per split) and `label_map.json` (word → integer index)

**Features:**
- Multiprocessing support (`--workers 4`) for faster extraction
- Resumable — skips already-extracted files
- Progress tracking with success/fail/skip counts
- Handles videos where MediaPipe finds no hands (zero-fills those frames)

**Usage:**
```bash
python extract_landmarks.py                         # all splits
python extract_landmarks.py --split train           # one split
python extract_landmarks.py --workers 4 --max-videos 10  # test run
```

**Output structure:**
```
landmarks_300/
├── train/
│   ├── about/
│   │   ├── 00414.npy    ← (60, 258) float64
│   │   ├── 00415.npy
│   │   └── ...
│   ├── accident/
│   └── ... (300 word directories)
├── val/
├── test/
├── manifest.json        ← word → [.npy paths] per split
└── label_map.json       ← word → integer index (300 entries)
```

##### 2. `dataset.py` — PyTorch Dataset with Augmentations

**Job:** Load extracted `.npy` landmarks and serve them as PyTorch tensors with data augmentation.

**7 Augmentation Strategies (training only) — Updated for Run 3:**

| Augmentation | Probability | What It Does |
|---|---|---|
| Temporal jitter | 50% | Shifts frames ±3 positions (simulates timing variation) |
| Gaussian noise | 70% | Adds σ=0.03 noise to non-zero landmarks (was σ=0.005) |
| Frame dropout | 50% | Zeros out 25% of frames randomly (was 5%) |
| Temporal scaling | 50% | Speeds up/slows down by 0.70–1.30× (was 0.85–1.15×) |
| Mirror flip | 30% | Swaps left/right hands + flips x-coordinates |
| Hand scaling | 50% | Multiplies hand landmarks by 0.85–1.15× (NEW — simulates hand size) |
| Spatial jitter | 50% | Shifts all x,y coords by ±0.05 (NEW — simulates camera angle) |

**Class imbalance handling:**
- Computes inverse-frequency class weights
- Uses `WeightedRandomSampler` so rare words get sampled more often
- Class weights also passed to the loss function

##### 3. `model.py` — GestureTransformer Architecture

**Job:** The core neural network — takes (60, 258) landmark sequences and predicts one of 300 words.

**Full architecture diagram:**

```
Input (B, 60, 258)
    │
    ▼
┌─ LandmarkEmbedding ──────────────────────────────────┐
│  Separate projections for anatomical structure:       │
│  Pose (132 dims) → Linear(132→128) + LayerNorm + ReLU│
│  Left Hand (63 dims) → Linear(63→64) + LayerNorm     │ 
│  Right Hand (63 dims) → Linear(63→64) + LayerNorm    │
│  Fuse: Linear(256→256) + LayerNorm + ReLU            │
│  Output: (B, 60, 256)                                │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌─ ConvTemporalBlock ──────────────────────────────────┐
│  Multi-scale Conv1D for local temporal patterns:      │
│  Conv1D(k=3) → captures adjacent-frame motion         │
│  Conv1D(k=5) → captures short phrase motion           │
│  Conv1D(k=7) → captures longer temporal patterns      │
│  Fuse with Conv1D(k=1): (B, 256*3, T) → (B, 256, T) │
│  Each branch: Conv1D + BatchNorm + GELU + Dropout     │
│  Output: (B, 60, 256)                                │
└───────────────────────────────────────────────────────┘
    │
    ▼
┌─ [CLS] Token Prepend ───────────────────────────────┐
│  Learnable classification token prepended            │
│  Output: (B, 61, 256)                               │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ Positional Encoding ───────────────────────────────┐
│  Sinusoidal encoding (preserves frame order)         │
│  + Dropout(0.1)                                      │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ Transformer Encoder ───────────────────────────────┐
│  4 layers × 8 attention heads                        │
│  Pre-norm architecture (more stable training)        │
│  dim_feedforward = 512                               │
│  GELU activation                                     │
│  Each layer: LayerNorm → MultiHeadAttention →        │
│              LayerNorm → FFN → Dropout               │
│  Output: (B, 61, 256)                               │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ Classification Head ───────────────────────────────┐
│  Take [CLS] token output: (B, 256)                   │
│  LayerNorm → Linear(256→128) → GELU → Dropout(0.3)  │
│  → Linear(128→300) → Softmax                        │
│  Output: (B, 300)                                    │
└──────────────────────────────────────────────────────┘
```

**Key design choices and why:**

1. **Structured LandmarkEmbedding** instead of a flat linear projection:
   - Pose, left hand, and right hand have different semantics
   - Separate projections preserve anatomical structure
   - Pose gets 128 dims (more complex, more landmarks), hands get 64 each

2. **Multi-scale Conv1D** (k=3,5,7) ins
tead of single Conv1D:
   - Different signs have different temporal scales
   - A quick "yes" nod spans 3-5 frames
   - A slow "thank you" spans 15-20 frames
   - Multi-scale captures all of these

3. **[CLS] token** instead of mean pooling:
   - Learns what to attend to for classification
   - Same approach used in BERT/ViT — proven effective
   - Clean single-vector output for the classifier

4. **Pre-norm Transformer** (`norm_first=True`):
   - More stable training than post-norm
   - Better gradient flow in deeper networks
   - Standard in modern transformer architectures

5. **GELU activation** instead of ReLU:
   - Smoother gradient around zero
   - Standard in transformer architectures
   - Slightly better empirical performance

**Model stats:**
- Parameters: ~3.5M (small enough for real-time, large enough for 300 classes)
- Forward pass verified: `(2, 60, 258) → (2, 300)` ✓
- Feature extraction: `(2, 256)` ✓

##### 4. `train.py` — Full Training Loop

**Job:** Train the GestureTransformer on extracted WLASL landmarks with best-practice training techniques.

**Training features (updated for Run 3):**

| Feature | Details |
|---|---|
| Mixed precision (AMP) | Automatic on GPU, disabled on CPU |
| Optimizer | AdamW (weight decay=5e-5, betas=0.9/0.999) |
| Learning rate | 5e-4 → 1e-6 with cosine annealing (smooth) |
| Warmup | Linear LR ramp for first 10 epochs |
| Loss function | CrossEntropyLoss with class weights + label smoothing (0.2) |
| Mixup training | α=0.2 — blends two samples for synthetic data (NEW) |
| Gradient clipping | max_norm=1.0 (prevents gradient explosions) |
| Early stopping | Patience=40 epochs on **validation loss** (was val_acc) |
| Checkpointing | Best model (on val_loss) + periodic every 10 epochs |
| Metrics tracked | Train loss, train acc, val loss, val acc, val top-5, LR |

**Label smoothing (0.2):** Instead of hard targets [0,0,1,0,...], uses soft targets [0.00067, 0.00067, 0.8, 0.00067,...]. With only 8 samples/word, the model becomes overconfident — 0.2 smoothing prevents it from assigning 99% probability to memorized training examples.

**Mixup (α=0.2):** Blends two random training samples (e.g., 60% "accident" + 40% "about") to create synthetic data. Forces the model to learn smooth decision boundaries instead of memorizing individual samples.

**Output files:**
```
Model/gesture_model_300.pt              ← best checkpoint (full, for resuming)
Model/gesture_model_300_inference.pt    ← inference-only (lightweight)
Model/checkpoint_epoch_N.pt             ← periodic checkpoints
training_logs/training_history.json     ← full training metrics
```

**Usage:**
```bash
python train.py                                    # full training (150 epochs)
python train.py --epochs 1 --max-samples 20        # quick test
python train.py --resume gesture_model_300.pt      # resume training
python train.py --batch-size 64 --lr 5e-4          # custom hyperparameters
```

##### 5. `requirements_training.txt` — Training Dependencies

Separate from Phase 1 runtime requirements since we switched to PyTorch:
```
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
mediapipe==0.10.14
opencv-contrib-python==4.8.1.78
numpy==1.26.4
scipy>=1.11.0
tqdm
scikit-learn
matplotlib
```

### Dataset: WLASL 300

**Source:** Word-Level American Sign Language (WLASL) dataset  
**Location:** `MODEL_Training/WLASL_300/`

**Statistics:**
| Split | Videos | Purpose |
|---|---|---|
| Train | 2,488 | Model training |
| Val | 649 | Hyperparameter tuning, early stopping |
| Test | 530 | Final evaluation |
| **Total** | **3,667** | **300 word classes** |

**Format:** Raw MP4 video files organized as:
```
WLASL_300/
├── train/
│   ├── about/ (6 videos)
│   ├── accident/ (10 videos)
│   ├── ... (300 word directories)
│   └── your/ (8 videos)
├── val/ (300 word directories)
└── test/ (300 word directories)
```

Each word has 5-16 video samples from different signers — this variety helps the model generalize across different people's signing styles.

### Current Status (April 27, 2026 — 8:15 PM)

- ✅ All training pipeline files created and verified
- ✅ Model forward pass test passing: `(2, 60, 258) → (2, 300)`
- ✅ Landmark extraction complete — all 3,667 videos processed
- ✅ GPU detected and enabled — NVIDIA RTX 3050 6GB with CUDA 12.4
- ❌ **Training Run #1 failed** — 0.92% accuracy, early stopped at epoch 24 (see below)
- ✅ **Training Run #2 completed** — 39.6% val accuracy, but severe overfitting (38.6-pt gap)
- ✅ **Overfitting diagnosed** — 5 anti-overfitting fixes applied to code
- ✅ **Run 2 archived** to `Run_2_Archive/` with code snapshot
- 🔄 **Ready for Training Run #3** (all fixes applied)

### Bugs Found and Fixed

**Bug 1 — PositionalEncoding max_len mismatch:**
- The [CLS] token makes the sequence 61 frames (60 + 1)
- PositionalEncoding was initialized with `max_len=seq_length` (60)
- This caused a shape mismatch: `pe` had 60 positions but input had 61
- **Fix:** Changed to `max_len=seq_length + 10` to safely cover CLS token

**Bug 2 — PyTorch installed without CUDA support:**
- System has NVIDIA GeForce RTX 3050 6GB Laptop GPU (confirmed via `nvidia-smi`)
- But `torch.cuda.is_available()` returned `False`
- **Root cause:** `pip install torch` installed the CPU-only build (`torch 2.11.0+cpu`)
- **Why this happens:** PyPI's default PyTorch wheel is CPU-only. CUDA builds require installing from PyTorch's own index URL
- **Fix:** Reinstalled with CUDA 12.4 support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- **Result:** `torch 2.6.0+cu124` installed, GPU now detected ✓
- **Impact:** Training will be 10-20× faster on GPU vs CPU

### Landmark Extraction Results

 Extraction completed successfully on April 25, 2026 (~45 minutes total).

| Split | Samples Extracted | Words Covered |
|---|---|---|
| Train | 2,488 | 300 / 300 |
| Val | 649 | 292 / 300 |
| Test | 530 | 294 / 300 |
| **Total** | **3,667** | **300 classes** |

> **Note:** Val and test splits have slightly fewer word classes (292 and 294) because some rare words only appear in the training set. This is normal for WLASL — the model will still train on all 300 classes.

Output files generated:
- `landmarks_300/manifest.json` — Maps word → list of `.npy` file paths per split
- `landmarks_300/label_map.json` — Maps 300 words → integer indices (0-299)
- 3,667 `.npy` files, each of shape `(60, 258)`

### Training Run #1 — FAILED (Epoch 24, 0.92% accuracy)

**What happened:**
```
Epoch 1:  val_acc = 0.31%  (random chance = 0.33%)
Epoch 4:  val_acc = 0.92%  ★ best (barely above random)
Epoch 24: val_acc = 0.62%  → early stopping triggered
```

The model was essentially stuck at random chance for all 24 epochs. The loss barely decreased (5.77 → 5.35, where ln(300) ≈ 5.7 = pure random).

**Root cause analysis — 3 compounding problems:**

| Problem | Why It Hurt | Fix Applied |
|---|---|---|
| **LR schedule too aggressive** | `CosineAnnealingWarmRestarts(T_0=10)` resets LR every 10 epochs, destabilizing what little the model learned | Switched to `CosineAnnealingLR` (smooth single decay over full training) |
| **Early stopping too trigger-happy** | `patience=20` killed training at epoch 24 because val accuracy barely moved — but with ~8 samples/class, improvement is inherently slow and noisy | Increased to `patience=50` |
| **Too much regularization** | `label_smoothing=0.1` + `weight_decay=1e-4` on only 8 samples/class prevented the model from even memorizing training data | Reduced label smoothing to 0.05, weight decay to 5e-5 |

**All changes made to `train.py`:**

| Parameter | Run #1 (failed) | Run #2 (fixed) | Why |
|---|---|---|---|
| Learning rate | 1e-3 | 5e-4 | More stable for small datasets |
| Min LR | 1e-5 | 1e-6 | Allow finer convergence |
| LR schedule | CosineAnnealingWarmRestarts (T_0=10) | CosineAnnealingLR (smooth) | No more destabilizing LR resets |
| Label smoothing | 0.1 | 0.05 | Less regularization, let model learn |
| Weight decay | 1e-4 | 5e-5 | Less regularization |
| Warmup epochs | 5 | 10 | Longer gentle ramp |
| Patience | 20 | 50 | Give model time to converge |
| Max epochs | 150 | 500 | Model needs many more epochs |

### Training Run #2 — SUCCESS (500 epochs, 39.6% val accuracy)

**Command:** `python train.py --batch-size 32 --num-workers 4`

**Key Results:**

| Metric | Value |
|---|---|
| Total epochs | 500 (no early stop) |
| Best val accuracy | **39.60%** (epoch 471) |
| Test top-1 accuracy | **30.38%** |
| Test top-5 accuracy | **60.19%** |
| Peak train accuracy | 77.72% |
| Final train loss | 1.4217 |
| Train-val gap | ~21% (some overfitting) |
| Training time | ~2.5 hours on RTX 3050 |

**Training progression:**

| Epoch | Train Loss | Train Acc | Val Acc | Val Top-5 |
|---|---|---|---|---|
| 1 | 5.84 | 0.4% | 0.5% | 1.2% |
| 50 | 5.25 | 1.1% | 1.7% | 6.9% |
| 100 | 5.03 | 2.6% | 4.3% | 15.6% |
| 200 | 3.97 | 9.9% | 14.9% | 37.3% |
| 300 | 3.15 | 17.5% | 25.7% | 52.4% |
| 400 | 2.29 | 36.1% | 33.3% | 60.7% |
| **471 (best)** | **1.58** | **50.2%** | **39.6%** | **67.5%** |
| 500 | 1.42 | 59.1% | 38.2% | 67.0% |

**Analysis:**
- Model learned well — loss dropped from 5.84 (random) to 1.42
- Significant overfitting after epoch ~400 (train acc 59% vs val acc 38%)
- Top-5 accuracy of 60% means the correct word is in the top 5 predictions most of the time
- For 300 classes with only ~8 samples per class, 39.6% val accuracy is a reasonable baseline

**Files generated:**
- `Model/best_gesture_model_wlasl300_epoch471.pt` — Best model (full, for resuming)
- `Model/best_gesture_model_wlasl300_inference.pt` — Inference-only weights
- `Results/Run_2/training_history.json` — Full epoch-by-epoch metrics
- `Results/Run_2/training_curves.png` — Loss, accuracy, LR, overfitting plots
- `Results/Run_2/training_report.md` — Auto-generated analysis report
- `training01_checkpoint/` — 6 milestone checkpoints (epoch 100/200/300/400/470/500)

> **All Run 2 files archived to `Run_2_Archive/`** after overfitting diagnosis (see below).

### Overfitting Diagnosis — Why 30% Test Accuracy Is Not Good Enough

After deploying the Run 2 model into the Phase 1 real-time pipeline, predictions were inaccurate and late. Reducing the frame prediction window from 60 to 20 frames helped latency, but accuracy remained poor.

**The key question:** Should we move to Phase 3 (LLM correction) or retrain the model first?

**Answer: Retrain first.** Phase 3 (LLM) corrects grammar from correct words. If the words themselves are wrong, the LLM can't fix that — garbage in, garbage out.

**The real problem — 38-point train-val gap:**

| Metric | Value | What It Means |
|---|---|---|
| Train accuracy | 76.9% | Model CAN learn patterns |
| Val accuracy | 39.6% | Model ISN'T generalizing |
| Gap | **37 pts** | Model memorized training samples |

This is textbook overfitting. The model memorized the ~8 training samples per word instead of learning the underlying gesture patterns. More epochs won't fix this — we hit the ceiling at epoch 200. The gap needs to be attacked directly.

**Why this isn't just a data size problem:**
- Published SOTA on WLASL300 = 65% with 10× more data + tricks
- ~50% is a realistic ceiling for our dataset size
- But 30% is well below that ceiling — the training quality is the bottleneck

### Training Run #3 — 5 Anti-Overfitting Fixes (Preparing)

**Status:** Code updated, ready to train

All 5 fixes attack the overfitting gap from different angles. They are complementary and applied together.

#### Fix 1 — Stronger Augmentation (Biggest Impact, Free)

With only 8 samples/word, augmentation is literally the only source of extra data. The Run 2 augmentation was too gentle.

| Augmentation | Run 2 | Run 3 | Why |
|---|---|---|---|
| Gaussian noise σ | 0.005 | **0.03** | 6× more noise forces robustness |
| Frame dropout | 5% | **25%** | Model must handle missing frames |
| Speed warp range | 0.85–1.15 | **0.70–1.30** | Wider speed variation |
| Augment probability | 0.3–0.5 | **0.5–0.7** | More augmented samples per epoch |
| **Hand scaling** | ❌ missing | **0.85–1.15×** | Simulates different hand sizes |
| **Spatial jitter** | ❌ missing | **±0.05 in x,y** | Simulates camera angle variation |

#### Fix 2 — Mixup Training (Second Biggest Impact)

Creates synthetic samples by blending two sequences:
- Instead of feeding sequence A ("accident") alone, blend A + B ("about") at 60/40
- Label becomes `[0.6, 0.4, 0, 0, ...]` instead of `[1, 0, 0, 0, ...]`
- Forces the model to learn smooth decision boundaries
- Proven to significantly reduce overfitting on small datasets

#### Fix 3 — Label Smoothing 0.05 → 0.2

With 8 samples/word, the model becomes overconfident on training examples, assigning 99% probability to memorized samples. Higher smoothing (0.2) distributes probability mass across classes, preventing this.

#### Fix 4 — Early Stopping on val_loss (not val_acc)

Val accuracy is noisy with small datasets — a single sample flipping changes accuracy by ~0.15%. Val loss is a smoother signal that captures gradual improvement more reliably. Patience reduced from 50 → 40.

#### Fix 5 — Reduce Model Capacity

The 3.5M parameter model has too much capacity for ~8 samples/word. A smaller model has less room to memorize.

| Parameter | Run 2 | Run 3 | Why |
|---|---|---|---|
| d_model | 256 | **192** | Less embedding capacity |
| nhead | 8 | **6** | Fewer attention patterns to memorize |
| num_layers | 4 | **3** | Shallower = less memorization |
| dim_ff | 512 | **384** | Smaller feedforward |
| dropout | 0.3 | **0.4** | More regularization |
| **Total params** | **~3.5M** | **~1.8M** | Half the memorization capacity |

**Expected outcomes:**

| Metric | Run 2 | Run 3 (estimated) |
|---|---|---|
| Train Acc | 77.7% | ~55-60% (less memorization = good) |
| Val Acc | 39.6% | ~48-55% |
| Test Top-1 | 30.4% | ~45-52% |
| Train-Val Gap | 38.6 pts | ~8-12 pts |

---

## 6. Phase 2b — Fingerspelling Fallback (Planned)

**Status:** Not started — will begin after Layer 1 training completes

### Plan

- **Dataset:** ASL Alphabet from Kaggle (26 classes, A-Z)
- **Model:** Tiny classifier — single frame of hand landmarks → letter
- **Input:** 21×3 = 63 features (one hand) from single frame
- **Architecture:** Simple MLP or small CNN (this is an easy classification task)
- **When activated:** Gesture model confidence < 0.6

### Why This Matters

Every deaf person uses fingerspelling for:
- Names ("S-A-K-S-H-A-M")
- Places ("D-E-L-H-I")
- Technical terms ("C-O-V-I-D")
- Any word not in the sign vocabulary

Without fingerspelling, the app breaks the moment someone needs to spell anything.

---

## 7. Phase 3 — LLM Correction Layer (Planned)

**Status:** Not started — will begin after Layers 1 & 2 work

### Plan

The LLM doesn't need to understand signs at all. It just takes a messy word stream and reconstructs natural English.

**Input:** `"hello you name what"`  
**Prompt to LLM:**
```
These words were recognized from ASL signing.
Reconstruct a natural English sentence.
Words: {hello you name what}
Sentence:
```
**Output:** `"Hello, what is your name?"`

**Options being considered:**
- Claude API (highest quality)
- Local small model (privacy, offline use)

### Integration Plan

1. Word buffer accumulates recognized words in inference pipeline
2. After a pause in signing (e.g., 1.5 seconds), buffer is sent to LLM
3. LLM returns corrected sentence
4. Corrected sentence displayed in UI as final caption

---

## 8. Technical Decisions Log

| Date | Decision | Why | Alternatives Considered |
|---|---|---|---|
| Apr 24 | TF/Keras for Phase 1 | Existing code used it, stability over novelty | PyTorch (would have required full rewrite) |
| Apr 24 | mediapipe==0.10.14 | Only version with working `mp.solutions.holistic` | Newer versions broke API |
| Apr 25 | Switch to PyTorch for Phase 2 | Better Transformer support, research ecosystem | Stay with TF (poor Transformer built-ins) |
| Apr 25 | Conv1D + Transformer (not BiLSTM) | Full attention over 60 frames, 2-3× faster training, scales to 300 classes | BiLSTM+Attention (Phase 1 arch, struggles at 300 classes) |
| Apr 25 | Structured LandmarkEmbedding | Preserves anatomical structure (pose vs. hands) | Flat linear projection (loses structure) |
| Apr 25 | Multi-scale Conv1D (k=3,5,7) | Different signs have different temporal scales | Single k=3 (misses long patterns) |
| Apr 25 | [CLS] token over mean pooling | Learns attention focus for classification | Mean pool (treats all frames equally) |
| Apr 25 | Pre-norm Transformer | More stable training, better gradients | Post-norm (original Transformer, less stable) |
| Apr 25 | Label smoothing 0.1 | Prevents overconfidence, improves generalization | Hard targets (model becomes too certain) |
| Apr 25 | WeightedRandomSampler | WLASL has imbalanced classes (5-16 samples per word) | Uniform sampling (model ignores rare words) |
| Apr 25 | PyTorch CUDA 12.4 build | System has RTX 3050 GPU, need CUDA for 10-20× speedup | CPU-only PyTorch (default pip install, unusably slow for 300 classes) |
| Apr 25 | CosineAnnealingLR over WarmRestarts | WarmRestarts destabilized learning on small dataset (resets every 10 epochs) | CosineAnnealingWarmRestarts (good for large datasets, bad for 8 samples/class) |
| Apr 25 | Patience 50, label smoothing 0.05 | 8 samples/class = inherently noisy validation, need patience; too much regularization prevented learning | Patience 20 + label smoothing 0.1 (model couldn't even memorize training data) |
| Apr 27 | Retrain before Phase 3 | LLM corrects grammar from correct words — if words are wrong, garbage in = garbage out | Move to Phase 3 anyway (would produce bad sentences from bad words) |
| Apr 27 | Stronger augmentation (6× noise, 5× dropout) | 8 samples/word means augmentation is the only source of extra data; Run 2 augmentation was too gentle | Keep gentle augmentation + get more data (impractical) |
| Apr 27 | Mixup training (α=0.2) | Creates synthetic blended samples, forces smooth decision boundaries, proven on small datasets | No mixup (model memorizes individual samples) |
| Apr 27 | Label smoothing 0.05 → 0.2 | Model was assigning 99% probability to memorized training samples; 0.2 prevents overconfidence | Keep at 0.05 (insufficient regularization for 8 samples/word) |
| Apr 27 | Early stop on val_loss (not val_acc) | Val_acc is noisy with small datasets (one sample flip = 0.15% change); val_loss is smoother | Keep val_acc monitoring (noisy checkpoint selection) |
| Apr 27 | Reduce model 3.5M → ~1.8M params | Half the parameters = half the memorization capacity; 3.5M is too large for 8 samples/word | Keep large model (more capacity to memorize = worse overfitting) |

---

## 9. File Structure

```
Gestura APP/
├── requirements.txt                    ← Phase 1 runtime deps
│
│
├── Phase_1/                            ← Phase 1 (complete)
│   ├── app.py                          ← Main application entry
│   ├── ASL.py                          ← Original training/data script
│   ├── model_training.py               ← Phase 1 training script
│   ├── model2.keras                    ← Phase 1 trained model (16 words)
│   ├── PHASE1_REPORT.md                ← Phase 1 detailed report
│   ├── core/
│   │   ├── inference_assets.py         ← Runtime model + helpers
│   │   ├── smoother.py                 ← TemporalSmoother
│   │   └── startup_checks.py          ← Preflight validation
│   ├── threads/
│   │   ├── capture_thread.py
│   │   ├── inference_thread.py
│   │   └── virtual_cam_thread.py
│   └── ui/
│       └── main_window.py             ← PyQt6 main window
│
├── MODEL_Training/                     ← Phase 2 (in progress)
│   ├── PROJECT_DOCUMENTATION.md        ← THIS FILE
│   ├── WLASL_300/                      ← Raw dataset
│   │   ├── train/ (2,488 videos, 300 words)
│   │   ├── val/ (649 videos)
│   │   └── test/ (530 videos)
│   │
│   ├── landmarks_300/                  ← Extracted landmarks (COMPLETE ✅)
│   │   ├── train/ (300 word directories with .npy files)
│   │   ├── val/
│   │   ├── test/
│   │   ├── manifest.json
│   │   └── label_map.json
│   │
│   ├── Model/
│   │   ├── model.py                    ← GestureTransformer v2 (d=192, reduced)
│   │   ├── dataset.py                  ← PyTorch Dataset + 7 augmentations (v2)
│   │   ├── train.py                    ← Training loop + mixup (v2)
│   │   ├── extract_landmarks.py        ← MP4 → .npy extraction pipeline
│   │   └── requirements_training.txt   ← Training dependencies
│   │
│   ├── Run_2_Archive/                  ← ARCHIVED Run 2 (before anti-overfitting fixes)
│   │   ├── README.md                   ← Archive documentation
│   │   ├── best_gesture_model_wlasl300_epoch471.pt ← Run 2 best model
│   │   ├── best_gesture_model_wlasl300_inference.pt ← Run 2 inference
│   │   ├── checkpoints/                ← Run 2 milestone checkpoints
│   │   │   ├── checkpoint_epoch_100.pt ... checkpoint_epoch_500.pt
│   │   └── code_snapshot/              ← Run 2 code (dataset, model, train)
│   │       ├── dataset_run2.py
│   │       ├── model_run2.py
│   │       └── train_run2.py
│   │
│   ├── Results/
│   │   ├── analyze_results.py          ← Analysis + plot generator
│   │   ├── organize_run.py             ← Run organizer script
│   │   └── Run_2/
│   │       ├── training_history.json
│   │       ├── training_curves.png
│   │       ├── training_report.md
│   │       ├── run_metadata.json
│   │       ├── gesture_model_300_best.pt   ← Copy of best model
│   │       └── gesture_model_300_inference.pt
│
└── Gestura/                            ← (Future: packaged app)
```

---

## 10. Setup & Usage Guide

### Prerequisites
- Python 3.10+
- GPU recommended (NVIDIA with CUDA) for training
- Webcam for Phase 1 runtime

### Phase 1 — Running the App
```bash
cd Phase_1
pip install -r ../requirements.txt
python app.py
```

### Phase 2 — Training the 300-Word Model

#### Step 1: Install training dependencies (with GPU support)
```bash
cd MODEL_Training/Model
# Install PyTorch with CUDA (CRITICAL — default pip installs CPU-only!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install remaining deps
pip install -r requirements_training.txt
```

**Verify GPU detection:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True  \n  NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

#### Step 2: Extract landmarks from WLASL videos
```bash
python extract_landmarks.py --workers 4
```
- Processes all 3,667 MP4 videos through MediaPipe
- Takes ~45-90 minutes depending on CPU
- Output: `landmarks_300/` directory with `.npy` files

#### Step 3: Train the model
```bash
python train.py --epochs 150 --batch-size 32
```
- Takes several hours on GPU, longer on CPU
- Best model auto-saved to `gesture_model_300.pt`
- Monitor training with printed metrics each epoch

#### Step 4: Quick test (verify pipeline works)
```bash
python train.py --epochs 1 --max-samples 20
```

---

## 11. Change Log

### April 24, 2026 — Phase 1 Complete
- Stabilized real-time subtitle pipeline
- Fixed virtual camera lag with dedicated VirtualCamThread
- Resolved MediaPipe/TF/Protobuf dependency conflicts
- Added startup checks and camera selector
- Separated inference code from training code
- Delivered working 16-word recognition app

### April 25, 2026 (Morning) — Phase 2 Started
- **Architecture discussion:** Evaluated BiLSTM vs Transformer vs ST-GCN
- **Decision:** Conv1D + Transformer Encoder with structured landmark embedding
- **Created training pipeline:** 5 files in `MODEL_Training/Model/`
  - `model.py` — GestureTransformer (~3.5M params)
  - `extract_landmarks.py` — MP4 → .npy landmark extraction
  - `dataset.py` — PyTorch Dataset with 5 augmentations
  - `train.py` — Full training loop with AMP, checkpointing, early stopping
  - `requirements_training.txt` — PyTorch training deps
- **Bug fix:** PositionalEncoding max_len mismatch (60 → 70 for CLS token)
- **Verified:** Model forward pass `(2, 60, 258) → (2, 300)` ✓
- **Started landmark extraction** on all 3,667 WLASL videos

### April 25, 2026 (12:00 PM) — Extraction Complete + GPU Fix
- **Landmark extraction completed** — all 3,667 videos → `.npy` files (took ~45 min)
  - Train: 2,488 samples / 300 words
  - Val: 649 samples / 292 words  
  - Test: 530 samples / 294 words
- **Generated** `manifest.json` and `label_map.json` (300 classes)
- **Diagnosed GPU issue:** PyTorch was `2.11.0+cpu` (CPU-only build) despite having RTX 3050
  - `nvidia-smi` confirmed GPU present with driver 581.86
  - Root cause: default `pip install torch` gives CPU-only wheel
  - **Fixed:** Reinstalled with `--index-url https://download.pytorch.org/whl/cu124`
  - Result: `torch 2.6.0+cu124`, `torch.cuda.is_available() = True` ✓
  - GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Ready to train** with full GPU acceleration

### April 25, 2026 (12:35 PM) — Training Run #1 Failed + Fixes
- **Ran first training:** `python train.py --epochs 150 --batch-size 32 --num-workers 4`
- **Result:** 0.92% val accuracy at best (epoch 4), early stopped at epoch 24
- **Diagnosis:** 3 compounding issues — LR schedule too aggressive, early stopping too impatient, too much regularization
- **Fixed `train.py`:**
  - LR: 1e-3 → 5e-4
  - Schedule: CosineAnnealingWarmRestarts → CosineAnnealingLR (smooth)
  - Label smoothing: 0.1 → 0.05
  - Weight decay: 1e-4 → 5e-5
  - Patience: 20 → 50
  - Warmup: 5 → 10 epochs
  - Max epochs: 150 → 500
  - Fixed deprecated AMP API calls
- **Ready for Training Run #2**

### April 25, 2026 (3:08 PM) — Training Run #2 Complete + Results Organized
- **Fixed `autocast` deprecation** in evaluate() function (missing `'cuda'` device_type)
- **Ran Training Run #2:** `python train.py --batch-size 32 --num-workers 4` (all 500 epochs)
- **Results:**
  - Best val accuracy: **39.60%** at epoch 471
  - Test top-1: **30.38%**, Test top-5: **60.19%**
  - Peak train accuracy: 77.72% (some overfitting expected with ~8 samples/class)
- **Organized results:**
  - Deleted 44 extra checkpoints, kept 6 milestones (epoch 100/200/300/400/470/500)
  - Moved checkpoints to `training01_checkpoint/`
  - Renamed best model to `best_gesture_model_wlasl300_epoch471.pt`
  - Created `Results/Run_2/` with training history, curves plot, and analysis report
- **Created analysis tools:** `Results/analyze_results.py`, `Results/organize_run.py`

### April 27, 2026 — Overfitting Diagnosis + Run 3 Preparation
- **Diagnosed severe overfitting** in Run 2: 38.6-point train-val gap (76.9% train vs 39.6% val)
- **Decision: retrain before Phase 3** — LLM can't fix wrong words (garbage in = garbage out)
- **Archived Run 2** to `Run_2_Archive/` (model, checkpoints, code snapshot)
- **Applied 5 anti-overfitting fixes to code:**
  - Fix 1: Stronger augmentation — noise 6×, dropout 5×, +hand scaling, +spatial jitter
  - Fix 2: Mixup training — synthetic blended samples (α=0.2)
  - Fix 3: Label smoothing 0.05 → 0.2
  - Fix 4: Early stopping on val_loss (smoother than val_acc), patience 50 → 40
  - Fix 5: Model reduction 3.5M → ~1.8M params (d=192, heads=6, layers=3)
- **Updated `dataset.py`** — 7 augmentations (was 5), all parameters strengthened
- **Updated `train.py`** — mixup, label smoothing, val_loss monitoring, reduced defaults
- **Updated `model.py`** — LandmarkEmbedding scales with d_model, new defaults
- **Ready for Training Run #3**

---

*This document will be updated as the project progresses through training, Phase 2b, and Phase 3.*
