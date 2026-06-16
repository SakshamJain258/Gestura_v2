"""
FingerspellingClassifier — lightweight MLP for ASL alphabet recognition.

Input:  63 floats (right-hand 21 landmarks × 3 coordinates from MediaPipe)
Output: 27 classes (A–Z + space)

Why so simple?
  - Single-frame classification, no temporal context needed
  - 63 input features → very small input space
  - Target hardware: CPU on a laptop during a call
  - Must run in < 5 ms per frame
"""

import torch
import torch.nn as nn


class FingerspellingClassifier(nn.Module):
    """3-layer MLP for single-frame hand landmark → letter classification."""

    def __init__(self, input_dim: int = 63, num_classes: int = 27, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 63) hand landmark features (normalized x,y,z per joint)
        Returns:
            logits: (batch, num_classes)
        """
        return self.net(x)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load model from a saved checkpoint dict."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_classes=ckpt.get("num_classes", 27),
            dropout=0.0,  # inference: no dropout
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model, ckpt.get("classes", [chr(i) for i in range(65, 91)] + [" "])
