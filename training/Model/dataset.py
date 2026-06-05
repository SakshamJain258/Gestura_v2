"""
PyTorch Dataset for WLASL 300 Landmarks — v2 (Anti-Overfitting)
================================================================
Loads pre-extracted .npy landmark files for training/validation/testing.

Each sample: (60, 258) float32 tensor + integer class label.

Augmentations (training only) — STRONGER for Run 3:
  - Temporal jitter: random shift ±3 frames
  - Landmark noise: Gaussian σ=0.03 (was 0.005)
  - Frame dropout: zero out 25% of frames (was 5%)
  - Time warp: speed variation 0.70–1.30× (was 0.85–1.15×)
  - Mirror flip: swap left/right hands
  - Hand scaling: multiply hand landmarks by 0.85–1.15 (NEW)
  - Spatial jitter: shift all landmarks ±0.05 in x,y (NEW)

Changes from Run 2 → Run 3:
  - Noise σ: 0.005 → 0.03
  - Frame dropout: 5% → 25%
  - Speed warp: 0.85–1.15 → 0.70–1.30
  - Augmentation probabilities: raised across the board
  - Added hand scaling augmentation (simulates different hand sizes)
  - Added spatial jitter augmentation (simulates camera angle variation)
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import uniform_filter1d


class WLASLDataset(Dataset):
    def __init__(self, landmarks_dir, split="train", augment=True, seq_length=60):
        """
        Args:
            landmarks_dir: Path to the landmarks_300/ directory
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation (True for train)
            seq_length: Number of frames per sample
        """
        self.seq_length = seq_length
        self.augment = augment and (split == "train")

        # Load manifest
        manifest_path = os.path.join(landmarks_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Load label map
        label_map_path = os.path.join(landmarks_dir, "label_map.json")
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        self.num_classes = len(self.label_map)

        # Build (path, label) pairs
        self.samples = []
        split_data = manifest.get(split, {})
        for word, npy_paths in split_data.items():
            label = self.label_map[word]
            for npy_path in npy_paths:
                self.samples.append((npy_path, label))

        # Compute class weights for imbalanced sampling
        class_counts = np.zeros(self.num_classes)
        for _, label in self.samples:
            class_counts[label] += 1
        class_counts = np.maximum(class_counts, 1)  # avoid div by zero
        self.class_weights = 1.0 / class_counts
        self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes

        # Sample weights for WeightedRandomSampler
        self.sample_weights = [self.class_weights[label] for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        landmarks = np.load(npy_path).astype(np.float32)  # (60, 258)

        # Ensure correct shape
        if landmarks.shape[0] < self.seq_length:
            pad = np.zeros((self.seq_length - landmarks.shape[0], landmarks.shape[1]), dtype=np.float32)
            landmarks = np.concatenate([landmarks, pad], axis=0)
        elif landmarks.shape[0] > self.seq_length:
            landmarks = landmarks[:self.seq_length]

        # Apply augmentation
        if self.augment:
            landmarks = self._augment(landmarks)

        return torch.from_numpy(landmarks), label

    def _augment(self, landmarks):
        """Apply random augmentations to landmark sequence.
        
        Run 3 changes: significantly stronger augmentation to combat
        overfitting with only ~8 samples/word. This is the single biggest
        lever for improving generalization without more data.
        """

        # 1. Temporal jitter: shift frames by ±3
        if random.random() < 0.5:
            shift = random.randint(-3, 3)
            landmarks = np.roll(landmarks, shift, axis=0)
            if shift > 0:
                landmarks[:shift] = 0.0
            elif shift < 0:
                landmarks[shift:] = 0.0

        # 2. Gaussian noise on landmarks (σ: 0.005 → 0.03)
        if random.random() < 0.7:
            noise = np.random.normal(0, 0.03, landmarks.shape).astype(np.float32)
            # Only add noise to non-zero frames (don't corrupt padding)
            mask = np.any(landmarks != 0, axis=1, keepdims=True)
            landmarks = landmarks + noise * mask

        # 3. Frame dropout: zero out ~25% of frames (was 5%)
        if random.random() < 0.5:
            n_drop = max(1, int(self.seq_length * 0.25))
            drop_indices = random.sample(range(self.seq_length), n_drop)
            landmarks[drop_indices] = 0.0

        # 4. Temporal scaling: speed variation 0.70–1.30× (was 0.85–1.15×)
        if random.random() < 0.5:
            scale = random.uniform(0.70, 1.30)
            orig_len = landmarks.shape[0]
            new_len = int(orig_len * scale)
            new_len = max(1, new_len)

            # Simple nearest-neighbor resampling
            indices = np.linspace(0, orig_len - 1, new_len).astype(int)
            scaled = landmarks[indices]

            # Pad/truncate back to seq_length
            if len(scaled) < self.seq_length:
                pad = np.zeros((self.seq_length - len(scaled), landmarks.shape[1]), dtype=np.float32)
                scaled = np.concatenate([scaled, pad], axis=0)
            else:
                scaled = scaled[:self.seq_length]

            landmarks = scaled

        # 5. Mirror augmentation: swap left/right hand landmarks
        if random.random() < 0.3:
            # pose: 0:132 (33*4), left_hand: 132:195 (21*3), right_hand: 195:258 (21*3)
            mirrored = landmarks.copy()
            lh = landmarks[:, 132:195].copy()
            rh = landmarks[:, 195:258].copy()
            mirrored[:, 132:195] = rh
            mirrored[:, 195:258] = lh

            # Flip x-coordinates for pose (every 4th starting at 0)
            for i in range(0, 132, 4):
                mirrored[:, i] = 1.0 - mirrored[:, i]
            # Flip x-coordinates for hands (every 3rd starting at 132/195)
            for start in [132, 195]:
                for i in range(start, start + 63, 3):
                    mirrored[:, i] = 1.0 - mirrored[:, i]

            landmarks = mirrored

        # 6. Hand scaling: simulate different hand sizes (NEW in Run 3)
        #    Multiplies hand landmarks by a random scale factor.
        #    This teaches the model to be invariant to signer hand size.
        if random.random() < 0.5:
            hand_scale = random.uniform(0.85, 1.15)
            # Only scale non-zero frames to preserve padding
            mask = np.any(landmarks[:, 132:258] != 0, axis=1, keepdims=True)
            landmarks[:, 132:258] = landmarks[:, 132:258] * (hand_scale * mask + (1 - mask))

        # 7. Spatial jitter: simulate camera angle variation (NEW in Run 3)
        #    Shifts all x,y coordinates by a small random offset.
        #    This teaches the model to be invariant to signer position in frame.
        if random.random() < 0.5:
            jitter_x = random.uniform(-0.05, 0.05)
            jitter_y = random.uniform(-0.05, 0.05)
            mask = np.any(landmarks != 0, axis=1).astype(np.float32)  # (T,)

            # Shift x-coordinates for pose (every 4th starting at 0)
            for i in range(0, 132, 4):
                landmarks[:, i] += jitter_x * mask
            # Shift y-coordinates for pose (every 4th starting at 1)
            for i in range(1, 132, 4):
                landmarks[:, i] += jitter_y * mask
            # Shift x-coordinates for hands (every 3rd starting at 132/195)
            for i in range(132, 258, 3):
                landmarks[:, i] += jitter_x * mask
            # Shift y-coordinates for hands (every 3rd starting at 133/196)
            for i in range(133, 258, 3):
                landmarks[:, i] += jitter_y * mask

        return landmarks

    def get_class_weights_tensor(self):
        """Returns class weight tensor for loss function."""
        return torch.FloatTensor(self.class_weights)


def get_dataloaders(landmarks_dir, batch_size=32, num_workers=4):
    """Create train/val/test DataLoaders with proper sampling."""
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = WLASLDataset(landmarks_dir, split="train", augment=True)
    val_ds = WLASLDataset(landmarks_dir, split="val", augment=False)
    test_ds = WLASLDataset(landmarks_dir, split="test", augment=False)

    # Weighted sampler for imbalanced classes
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds.num_classes, train_ds.class_weights
