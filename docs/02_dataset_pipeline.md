# 02 — Dataset Pipeline

## Dataset: WLASL-300

The **Word-Level American Sign Language (WLASL)** dataset is the largest publicly available ASL video dataset, created by Dongxu Li et al. (2020). Gestura v2 uses the 300-word subset (WLASL-300).

| Property | Value |
|---|---|
| Source | [WLASL GitHub Repository](https://github.com/dxli94/WLASL) |
| Total word classes | 300 |
| Total videos used | ~3,667 |
| Avg videos per class | ~8–12 |
| Video format | MP4 |
| Split | Train / Val / Test (standard WLASL split) |

> **Note:** The raw video dataset (~18 GB) is NOT included in this repository. It must be downloaded separately and placed in the `WLASL_300/` directory before running landmark extraction.

---

## Why Landmark-Based Instead of Raw Video?

Instead of training on raw RGB video frames (pixels), Gestura converts every video into a sequence of **MediaPipe skeletal landmarks**. Each frame becomes 258 floating-point numbers describing the 3D positions of pose and hand joints.

**Advantages:**
- **Compact** — 258 floats/frame vs. 640×480×3 = 921,600 pixels/frame
- **Background invariant** — the model never sees clothing, skin tone, or lighting
- **Signer invariant** — landmarks normalize for position and scale across different people
- **Real-time capable** — inference on CPU is fast enough for live use
- **Interpretable** — you can visualize exactly what the model sees by drawing the skeleton

---

## Feature Vector: 258 Dimensions

Each frame is converted to a `(258,)` float32 vector:

```
[0:132]   — Pose landmarks:      33 joints × (x, y, z, visibility) = 132
[132:195] — Left hand landmarks: 21 joints × (x, y, z)             = 63
[195:258] — Right hand landmarks:21 joints × (x, y, z)             = 63
```

If a hand is not detected in a frame (e.g., it left the frame), that segment is filled with zeros. The model was trained with this behavior, so zeros indicate "hand not visible" — a valid and meaningful signal.

---

## Sequence Shape

Each video sample is represented as:

```
(60 frames, 258 features)  →  shape: (60, 258)  dtype: float32
```

Videos shorter than 60 frames are **zero-padded at the end**. Videos longer than 60 frames are **truncated to the first 60 frames**.

> **Why 60 frames?** At 25 fps, 60 frames = ~2.4 seconds. This covers the full duration of most WLASL word signs, which typically last 0.5–2 seconds.

---

## Landmark Extraction Pipeline

**Script:** `training/Model/extract_landmarks.py`

### How to run

```powershell
cd training/Model
python extract_landmarks.py --workers 4
```

### Process per video

```
1. Open video with cv2.VideoCapture()
2. For each frame:
   a. Convert BGR → RGB
   b. Run MediaPipe Holistic (model_complexity=1)
   c. Extract pose, left-hand, right-hand landmark coordinates
   d. Concatenate into 258-dim vector
3. Pad or truncate to 60 frames
4. Save as .npy file → landmarks_300/{split}/{word}/{video_id}.npy
```

### Output structure

```
landmarks_300/
├── train/
│   ├── book/
│   │   ├── 00123.npy   # shape: (60, 258)
│   │   └── 00456.npy
│   ├── drink/
│   └── ...
├── val/
├── test/
├── manifest.json    # maps word → list of .npy paths per split
└── label_map.json   # maps word → integer class index (sorted alphabetically)
```

### label_map.json

Maps each word to its integer class index:

```json
{
  "about": 0,
  "accident": 1,
  "africa": 2,
  ...
  "your": 299
}
```

Classes are sorted alphabetically, so `about=0`, `accident=1`, ..., `your=299`.

### manifest.json

```json
{
  "train": {
    "book": ["landmarks_300/train/book/00123.npy", ...],
    "drink": [...]
  },
  "val": { ... },
  "test": { ... }
}
```

---

## MediaPipe Configuration

All landmark extraction (both during training and at inference) uses identical MediaPipe Holistic settings to ensure consistency:

```python
mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,        # matches training exactly
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

> **Critical:** `model_complexity=1` must be the same during both training (extraction) and live inference. Using a different complexity level produces slightly different landmark coordinate distributions, which degrades model accuracy.

---

## Class Imbalance

WLASL-300 is heavily imbalanced — popular words have 20+ videos while rare words may have only 5. The training pipeline addresses this with:

1. **Class-weighted cross-entropy loss** — rare class errors are penalized more
2. **WeightedRandomSampler** — oversamples rare classes during training batches

---

## Fingerspelling Dataset

For the fingerspelling model (A–Z classifier), a **synthetic landmark dataset** was generated using `training/fingerspelling/generate_synthetic_landmarks.py`. This script creates plausible hand landmark positions for each letter with random variation (jitter, scale, rotation) to simulate real-world diversity without requiring a dedicated video collection effort.
