# Run 2 - GestureTransformer WLASL300

Archived: April 27, 2026

This folder contains only Run 2 artifacts. Do not mix these files with Run 3 artifacts; the model architecture and training setup are different.

## Model

| Item | Value |
|---|---|
| Model | GestureTransformer |
| Dataset | WLASL 300 |
| Parameters | 3,455,532 |
| Total epochs | 500 |
| Best epoch | 471 |
| Best validation accuracy | 39.60% |
| Test top-1 accuracy | 30.38% |
| Test top-5 accuracy | 60.19% |
| Main issue | Overfitting, 38.61-point train-val gap |

## Contents

```text
Run_2_GestureTransformer_WLASL300_3p46M_Val39p60/
|-- best_gesture_model_wlasl300_epoch471.pt
|-- best_gesture_model_wlasl300_inference.pt
|-- checkpoints/
|   |-- checkpoint_epoch_100.pt
|   |-- checkpoint_epoch_200.pt
|   |-- checkpoint_epoch_300.pt
|   |-- checkpoint_epoch_400.pt
|   |-- checkpoint_epoch_470.pt
|   `-- checkpoint_epoch_500.pt
|-- code_snapshot/
|   |-- dataset_run2.py
|   |-- model_run2.py
|   `-- train_run2.py
|-- training_logs/
|   `-- logs_prev.json
`-- results/
    `-- Run_2/
        |-- gesture_model_300_best.pt
        |-- gesture_model_300_inference.pt
        |-- run_metadata.json
        |-- training_curves.png
        |-- training_history.json
        `-- training_report.md
```

## Load Example

```python
import torch
from code_snapshot.model_run2 import GestureTransformer

model = GestureTransformer(
    num_classes=300,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_ff=512,
    dropout=0.3,
)
checkpoint = torch.load("best_gesture_model_wlasl300_epoch471.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
```
