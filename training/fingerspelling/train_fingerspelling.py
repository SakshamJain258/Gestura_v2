"""
Fingerspelling Training Script
==============================

Trains FingerspellingClassifier on ASL alphabet hand landmark data.

Data source options (tries in order):
  1. Kaggle ASL Alphabet dataset (images) → extract landmarks with MediaPipe
  2. Pre-extracted .npy landmark files in fingerspelling_landmarks/

Usage:
    # Extract landmarks from image dataset first:
    python train_fingerspelling.py --extract --data-dir /path/to/asl_alphabet_train

    # Train on already-extracted landmarks:
    python train_fingerspelling.py --landmarks-dir ./fingerspelling_landmarks

    # Quick test run:
    python train_fingerspelling.py --epochs 5 --landmarks-dir ./fingerspelling_landmarks

Output:
    training/Results/Fingerspelling/fingerspelling_model.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
APP_DIR = PROJECT_ROOT / "app"
RESULTS_DIR = PROJECT_ROOT / "training" / "Results" / "Fingerspelling"
LANDMARKS_DIR = SCRIPT_DIR / "fingerspelling_landmarks"

# Make sure app modules are importable for the FingerspellingClassifier
sys.path.insert(0, str(APP_DIR))


# ── ASL classes ────────────────────────────────────────────────────────────────
CLASSES = [chr(i) for i in range(65, 91)] + ["space"]  # A-Z + space
# Map folder names to class indices
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
CLASS_TO_IDX["space"] = 26   # some datasets use 'space' or 'nothing'


# ── Dataset ───────────────────────────────────────────────────────────────────

class FingerspellingDataset(Dataset):
    """Dataset loaded from pre-extracted .npy landmark files."""

    def __init__(self, landmarks_dir: Path, augment: bool = False):
        self.samples = []
        self.labels = []
        self.augment = augment

        for class_name in CLASSES:
            class_dir = landmarks_dir / class_name
            if not class_dir.exists():
                print(f"  WARNING: No data dir for class '{class_name}'")
                continue

            class_idx = CLASS_TO_IDX[class_name]
            for npy_path in sorted(class_dir.glob("*.npy")):
                data = np.load(npy_path).astype(np.float32)
                if data.shape != (63,):
                    continue   # skip malformed
                self.samples.append(data)
                self.labels.append(class_idx)

        print(f"  Loaded {len(self.samples)} samples across {len(CLASSES)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()
        y = self.labels[idx]

        if self.augment:
            # Gaussian noise on landmark positions
            x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            x *= scale

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ── Landmark Extraction ───────────────────────────────────────────────────────

def extract_landmarks_from_images(data_dir: Path, output_dir: Path, max_samples: int = None):
    """Extract MediaPipe right-hand landmarks from ASL alphabet image dataset."""
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting landmarks from: {data_dir}")
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.upper()
            if class_name not in CLASS_TO_IDX and class_name.lower() not in CLASS_TO_IDX:
                print(f"  Skipping unknown class: {class_dir.name}")
                continue

            class_name_key = class_name if class_name in CLASS_TO_IDX else class_name.lower()
            out_class_dir = output_dir / class_name_key
            out_class_dir.mkdir(parents=True, exist_ok=True)

            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if max_samples is not None:
                image_files = sorted(image_files)[:max_samples]
            saved = 0
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    landmarks = np.array(
                        [[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32
                    ).flatten()
                    out_path = out_class_dir / f"{img_path.stem}.npy"
                    np.save(str(out_path), landmarks)
                    saved += 1

            print(f"  {class_name_key}: {saved}/{len(image_files)} extracted")

    print("Extraction complete.")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    from core.fingerspelling_model import FingerspellingClassifier

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    landmarks_dir = Path(args.landmarks_dir)

    if not landmarks_dir.exists():
        print(f"ERROR: Landmarks directory not found: {landmarks_dir}")
        print("Run with --extract first to extract landmarks from images.")
        sys.exit(1)

    print(f"Loading dataset from: {landmarks_dir}")
    dataset = FingerspellingDataset(landmarks_dir, augment=True)

    if len(dataset) == 0:
        print("ERROR: No samples found. Check your landmarks directory.")
        sys.exit(1)

    # Train/val split
    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FingerspellingClassifier(num_classes=len(CLASSES), dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()

        t_acc = train_correct / train_size * 100
        v_acc = val_correct / val_size * 100
        t_loss = train_loss / train_size
        v_loss = val_loss / val_size

        history.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                         "train_acc": t_acc, "val_acc": v_acc})

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train {t_acc:.1f}%  val {v_acc:.1f}%  "
                  f"loss {t_loss:.4f}/{v_loss:.4f}")

        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": len(CLASSES),
                "classes": CLASSES,
                "val_acc": v_acc,
                "epoch": epoch,
            }
            torch.save(checkpoint, RESULTS_DIR / "fingerspelling_model.pt")

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {RESULTS_DIR / 'fingerspelling_model.pt'}")

    # Save history
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Gestura fingerspelling classifier")
    parser.add_argument("--extract", action="store_true",
                        help="Extract landmarks from image dataset first")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Path to ASL alphabet image dataset (for --extract)")
    parser.add_argument("--landmarks-dir", type=str,
                        default=str(LANDMARKS_DIR),
                        help="Directory of pre-extracted .npy landmark files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max image samples to extract per class (to speed up extraction)")
    args = parser.parse_args()

    if args.extract:
        if not args.data_dir:
            print("ERROR: --data-dir required with --extract")
            sys.exit(1)
        extract_landmarks_from_images(
            Path(args.data_dir),
            Path(args.landmarks_dir),
            max_samples=args.max_samples,
        )

    train(args)


if __name__ == "__main__":
    main()
