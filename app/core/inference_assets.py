"""
Runtime inference assets shared by the desktop app.

This file contains only inference-time pieces:
- MediaPipe helpers
- label actions
- model architecture used at inference

It intentionally excludes data collection and training utilities.
"""

import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


LABEL_MAP_PATH = (
    Path(__file__).resolve().parents[2] / "MODEL_Training" / "landmarks_300" / "label_map.json"
)


def _load_actions_from_label_map(path: Path = LABEL_MAP_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    actions = [word for word, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]
    return np.array(actions)


actions = _load_actions_from_label_map()


class PositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LandmarkEmbedding(nn.Module):
    def __init__(self, d_model=256, dropout=0.3):
        super().__init__()
        self.pose_proj = nn.Sequential(
            nn.Linear(33 * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lh_proj = nn.Sequential(
            nn.Linear(21 * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rh_proj = nn.Sequential(
            nn.Linear(21 * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64 + 64, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        pose = x[:, :, :132]
        lh = x[:, :, 132:195]
        rh = x[:, :, 195:258]

        pose_emb = self.pose_proj(pose)
        lh_emb = self.lh_proj(lh)
        rh_emb = self.rh_proj(rh)
        fused = torch.cat([pose_emb, lh_emb, rh_emb], dim=-1)
        return self.fuse(fused)


class ConvTemporalBlock(nn.Module):
    def __init__(self, d_model=256, dropout=0.3):
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
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        c1 = self.conv1(x_t)
        c2 = self.conv2(x_t)
        c3 = self.conv3(x_t)
        multi = torch.cat([c1, c2, c3], dim=1)
        out = self.fuse(multi)
        return out.transpose(1, 2)


class GestureTransformer(nn.Module):
    def __init__(
        self,
        num_classes=300,
        input_dim=258,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=512,
        dropout=0.3,
        seq_length=60,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.landmark_embed = LandmarkEmbedding(d_model, dropout)
        self.conv_temporal = ConvTemporalBlock(d_model, dropout)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.landmark_embed(x)
        x = self.conv_temporal(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)

        cls_out = x[:, 0]
        return self.classifier(cls_out)


def build_gesture_model(
    num_classes=300,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_ff=512,
    dropout=0.3,
    seq_length=60,
):
    return GestureTransformer(
        num_classes=num_classes,
        input_dim=258,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        seq_length=seq_length,
    )


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmark(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(10, 255, 10), thickness=5, circle_radius=10),
        mp_drawing.DrawingSpec(color=(10, 2, 130), thickness=5, circle_radius=10),
    )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])

