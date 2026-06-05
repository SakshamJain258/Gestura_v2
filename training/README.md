# Trained Models

Each trained model run is stored in its own folder. The files are intentionally not merged because Run 2 and Run 3 use different training settings and model capacities.

## Runs

| Folder | Model | Params | Best val acc | Notes |
|---|---|---:|---:|---|
| `Run_2_GestureTransformer_WLASL300_3p46M_Val39p60` | GestureTransformer WLASL300 | 3,455,532 | 39.60% | Higher accuracy, severe overfitting |
| `Run_3_GestureTransformer_WLASL300_AntiOverfitting_1p66M_Val6p78` | GestureTransformer WLASL300 anti-overfitting | 1,661,868 | 6.78% | Reduced model with stronger regularization |

## Shared Data And Active Code

- `../WLASL_300/` keeps the original video dataset.
- `../landmarks_300/` keeps the extracted landmark dataset.
- `../Model/` keeps the active training/extraction code only.
- `../Results/` keeps reusable result-analysis scripts.
