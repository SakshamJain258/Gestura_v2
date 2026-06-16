# 03 — Model Architecture

## Overview

Gestura v2 uses two neural network models:

1. **GestureTransformer** — word-level ASL recognition (300 WLASL classes)
2. **FingerspellingClassifier** — single-frame A–Z letter recognition

Both are implemented in PyTorch and defined in `app/core/inference_assets.py` (inference copy) and `training/Model/model.py` (training copy with additional utilities).

---

## 1. GestureTransformer

### Motivation

Why a Transformer instead of an LSTM (as used in v1)?

- **Global temporal attention:** Transformers directly compare any two frames in the sequence. An LSTM propagates information step-by-step and struggles with long-range dependencies.
- **Parallelism:** Transformer training is parallelizable across time; LSTMs are sequential.
- **Scaling:** Transformers scale more smoothly to larger class counts (300 vs 17 in v1).
- **CLS token pattern:** A learnable classification token gives a clean summary representation for the final linear head.

### Input / Output

```
Input:  (batch_size, 60, 258)   — batch of 60-frame landmark sequences
Output: (batch_size, 300)       — logits over 300 ASL word classes
```

### Architecture Flow

```
Input (B, 60, 258)
    │
    ▼ LandmarkEmbedding
    │  ├─ Pose  (B, 60, 132) → Linear(132, 96)  → LayerNorm → ReLU → Dropout
    │  ├─ LHand (B, 60,  63) → Linear(63,  48)  → LayerNorm → ReLU → Dropout
    │  └─ RHand (B, 60,  63) → Linear(63,  48)  → LayerNorm → ReLU → Dropout
    │  └─ cat(96+48+48=192) → Linear(192, 192)  → LayerNorm → ReLU → Dropout
    │
    ▼ (B, 60, 192)
    │
    ▼ ConvTemporalBlock (multi-scale)
    │  ├─ Conv1D(192, 192, k=3, pad=1) → BN → GELU → Dropout
    │  ├─ Conv1D(192, 192, k=5, pad=2) → BN → GELU → Dropout
    │  └─ Conv1D(192, 192, k=7, pad=3) → BN → GELU → Dropout
    │  └─ cat(dim=channel, 576) → Conv1D(576, 192, k=1) → BN → GELU → Dropout
    │
    ▼ (B, 60, 192)
    │
    ▼ Prepend CLS token → (B, 61, 192)
    │
    ▼ Sinusoidal PositionalEncoding (additive)
    │
    ▼ TransformerEncoder × 3 layers (norm_first=True, batch_first=True)
    │  Each layer: MultiHeadSelfAttention(heads=6) + FFN(dim=384, GELU)
    │
    ▼ (B, 61, 192)
    │
    ▼ CLS token slice → x[:, 0] → (B, 192)
    │
    ▼ Classifier head:
    │  LayerNorm(192) → Linear(192, 96) → GELU → Dropout → Linear(96, 300)
    │
    ▼ Output: (B, 300) logits
```

### Component Details

#### LandmarkEmbedding

Pose and hands are projected **separately** before fusing. This preserves anatomical structure — the model learns body state and hand shapes independently, then combines them.

| Projector | Input | Output |
|---|---|---|
| Pose | 132 (33×4) | 96 (d_model/2) |
| Left hand | 63 (21×3) | 48 (d_model/4) |
| Right hand | 63 (21×3) | 48 (d_model/4) |
| Fusion | 192 | 192 |

#### ConvTemporalBlock (Multi-scale)

Three parallel Conv1D branches with kernel sizes 3, 5, and 7 capture motion patterns at different timescales:
- **k=3:** Short motion — finger flick, quick wrist snap
- **k=5:** Medium motion — hand arc, shoulder dip
- **k=7:** Longer motion — arm extension, two-beat signs

All outputs are concatenated on the channel dimension and fused with a 1×1 conv. Applied **before** the Transformer to give it temporally-enriched features.

#### Positional Encoding

Sinusoidal positional encoding (from the original "Attention is All You Need" paper) added after prepending the CLS token. Essential because Transformers have no inherent sequence order — without it, frame 1 is indistinguishable from frame 60.

#### Transformer Encoder (Pre-LayerNorm)

PyTorch `nn.TransformerEncoderLayer` with:
- `norm_first=True` — Pre-LayerNorm: more stable gradient flow (from "On Layer Normalization" paper)
- `batch_first=True` — (batch, seq, features) convention
- `activation="gelu"` — smoother than ReLU for Transformer contexts

### Hyperparameters (Run 3 — current production)

| Parameter | Value | Changed from Run 2? |
|---|---|---|
| `d_model` | 192 | ↓ from 256 |
| `nhead` | 6 | ↓ from 8 |
| `num_layers` | 3 | ↓ from 4 |
| `dim_ff` | 384 | ↓ from 512 |
| `dropout` (train) | 0.4 | same |
| `dropout` (infer) | 0.0 | same |
| Total parameters | ~1.66M | ↓ from ~3.8M |

The size reduction from Run 2 → Run 3 was intentional: smaller model = less overfitting on a small dataset.

---

## 2. FingerspellingClassifier

A lightweight 3-layer MLP predicting a single A–Z letter (or space) from one frame's right-hand landmarks.

### Why single-frame?

Fingerspelling letters are mostly **static hand shapes**, not motion sequences. Single-frame inference means the letter appears instantly on screen without the 20–60 frame accumulation delay of the GestureTransformer.

### Input / Output

```
Input:  (batch_size, 63)   — right-hand 21 landmarks × (x, y, z)
Output: (batch_size, 27)   — logits over A–Z + space (27 classes)
```

### Architecture

```
Linear(63 → 256) → ReLU → Dropout(0.3)
Linear(256 → 128) → ReLU → Dropout(0.3)
Linear(128 → 27)
```

### Right-hand extraction at inference

```python
# From the full 258-dim keypoint vector:
rh_keypoints = keypoints[195:258]  # indices 195–257 = right hand
```

---

## Inference vs. Training Model Files

| File | Contents | Size |
|---|---|---|
| `gesture_model_300_best.pt` | Full checkpoint: weights + optimizer + metadata | ~20 MB |
| `gesture_model_300_inference.pt` | State dict only (no optimizer) | ~6.7 MB |

The app loads the `_inference.pt` version. The `_best.pt` is only needed to resume training from the best epoch.
