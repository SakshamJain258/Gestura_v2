    """
Gesture Transformer Model — Layer 1 (v2, reduced capacity)
============================================================
Conv1D feature extractor + Transformer Encoder for 300-class
sign language word recognition from landmark sequences.

Architecture:
    Input (B, 60, 258)
    → Conv1D Feature Extractor (local temporal patterns)
    → Positional Encoding (frame order information)
    → Transformer Encoder (global temporal attention)
    → Temporal Pooling
    → Classification Head
    → Output (B, 300)

Changes from Run 2 → Run 3 (Fix 5 — Reduce Model Capacity):
    - LandmarkEmbedding now scales internal dims proportional to d_model
    - Default d_model: 256 → 192 (set via train.py CLI args)
    - Default nhead: 8 → 6
    - Default num_layers: 4 → 3
    - Default dim_ff: 512 → 384
    - Default dropout: 0.3 → 0.4
    - Estimated params: ~3.5M → ~1.8M
    
    A smaller model has less capacity to memorize training samples,
    which directly reduces the 38-point train-val gap.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LandmarkEmbedding(nn.Module):
    """
    Structured embedding that processes pose, left-hand, and right-hand
    landmarks separately before fusing them.
    This preserves anatomical structure information.

    Run 3 change: internal dimensions now scale proportionally to d_model
    instead of being hardcoded. This allows the embedding to work correctly
    when d_model is reduced from 256 to 192 (or any other value).
    """

    def __init__(self, d_model=192, dropout=0.4):
        super().__init__()
        # Scale internal dims proportional to d_model
        # At d_model=256: pose=128, hand=64 (original)
        # At d_model=192: pose=96,  hand=48
        pose_dim = d_model // 2
        hand_dim = d_model // 4

        # Separate projections for each body part
        self.pose_proj = nn.Sequential(
            nn.Linear(33 * 4, pose_dim),   # pose: 132 dims
            nn.LayerNorm(pose_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lh_proj = nn.Sequential(
            nn.Linear(21 * 3, hand_dim),    # left hand: 63 dims
            nn.LayerNorm(hand_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rh_proj = nn.Sequential(
            nn.Linear(21 * 3, hand_dim),    # right hand: 63 dims
            nn.LayerNorm(hand_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Fuse to d_model
        fuse_input = pose_dim + hand_dim + hand_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_input, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, 258)
        pose = x[:, :, :132]       # (B, T, 132)
        lh   = x[:, :, 132:195]    # (B, T, 63)
        rh   = x[:, :, 195:258]    # (B, T, 63)

        pose_emb = self.pose_proj(pose)
        lh_emb   = self.lh_proj(lh)
        rh_emb   = self.rh_proj(rh)

        fused = torch.cat([pose_emb, lh_emb, rh_emb], dim=-1)
        return self.fuse(fused)  # (B, T, d_model)


class ConvTemporalBlock(nn.Module):
    """Multi-scale Conv1D for local temporal pattern extraction."""

    def __init__(self, d_model=192, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Fuse multi-scale features
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, d_model) → transpose for Conv1D
        x_t = x.transpose(1, 2)  # (B, d_model, T)

        c1 = self.conv1(x_t)     # (B, d_model, T)
        c2 = self.conv2(x_t)     # (B, d_model, T)
        c3 = self.conv3(x_t)     # (B, d_model, T)

        multi = torch.cat([c1, c2, c3], dim=1)  # (B, d_model*3, T)
        out = self.fuse(multi)                    # (B, d_model, T)

        return out.transpose(1, 2)  # (B, T, d_model)


class GestureTransformer(nn.Module):
    """
    Conv1D + Transformer Encoder for sign language gesture recognition.

    Args:
        num_classes: Number of output classes (300 for WLASL-300)
        input_dim:   Input feature dimension (258 for pose+hands)
        d_model:     Transformer hidden dimension
        nhead:       Number of attention heads
        num_layers:  Number of Transformer encoder layers
        dim_ff:      Feedforward network dimension
        dropout:     Dropout rate
        seq_length:  Input sequence length (frames)
    """

    def __init__(
        self,
        num_classes=300,
        input_dim=258,
        d_model=192,
        nhead=6,
        num_layers=3,
        dim_ff=384,
        dropout=0.4,
        seq_length=60,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # 1. Structured landmark embedding (now scales with d_model)
        self.landmark_embed = LandmarkEmbedding(d_model, dropout)

        # 2. Multi-scale Conv1D for local temporal features
        self.conv_temporal = ConvTemporalBlock(d_model, dropout)

        # 3. Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length + 10, dropout=dropout)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 5. Learnable [CLS] token for classification (alternative to mean pooling)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 6. Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for transformer layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, T, 258) landmark sequences
            return_features: If True, also return pre-classifier features

        Returns:
            logits: (B, num_classes)
            features: (B, d_model) — only if return_features=True
        """
        B = x.size(0)

        # Structured embedding: (B, T, 258) → (B, T, d_model)
        x = self.landmark_embed(x)

        # Multi-scale Conv1D: capture local temporal patterns
        x = self.conv_temporal(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)           # (B, T+1, d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (B, T+1, d_model)

        # Use [CLS] token output for classification
        cls_out = x[:, 0]  # (B, d_model)

        # Classifier
        logits = self.classifier(cls_out)  # (B, num_classes)

        if return_features:
            return logits, cls_out
        return logits

    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization/debugging.
        Note: requires hooks or manual computation — placeholder for now.
        """
        # TODO: implement attention extraction for interpretability
        pass


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def model_summary(num_classes=300):
    """Print a summary of the model."""
    model = GestureTransformer(num_classes=num_classes)
    n_params = count_parameters(model)

    print(f"{'='*60}")
    print(f"  GestureTransformer v2 — {num_classes} classes")
    print(f"{'='*60}")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Architecture:")
    print(f"    LandmarkEmbedding → (B, 60, {model.d_model})")
    print(f"    ConvTemporalBlock → (B, 60, {model.d_model})  [multi-scale k=3,5,7]")
    print(f"    + [CLS] token     → (B, 61, {model.d_model})")
    print(f"    PositionalEncoding")
    print(f"    TransformerEncoder → 3 layers × 6 heads")
    print(f"    Classifier         → {num_classes} classes")
    print(f"{'='*60}")

    # Quick forward pass test
    dummy = torch.randn(2, 60, 258)
    out = model(dummy)
    print(f"  Forward pass: input {dummy.shape} → output {out.shape}")
    print(f"  ✓ Model OK")

    return model


if __name__ == "__main__":
    model_summary(300)
