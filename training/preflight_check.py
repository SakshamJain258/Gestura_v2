# -*- coding: utf-8 -*-
"""
Run 3 Pre-flight Check
======================
Run this before starting training to verify:
  1. PyTorch version and CUDA status
  2. landmarks_300 directory and data integrity
  3. Model forward pass
  4. DataLoader (one batch)
  5. Mixup sanity check

Usage:
    python preflight_check.py
    python preflight_check.py --landmarks-dir "C:/path/to/landmarks_300"
"""

import os
import sys
import json
import argparse
import traceback

# Force UTF-8 output so Unicode chars don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(SCRIPT_DIR, "Model")
DEFAULT_LM_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "landmarks_300")

# Add Model dir to path so we can import model.py / dataset.py
sys.path.insert(0, MODEL_DIR)

PASS  = "  [PASS] "
FAIL  = "  [FAIL] "
WARN  = "  [WARN] "
INFO  = "  [INFO] "


def check_pytorch():
    """Check PyTorch install + CUDA."""
    print("--- 1. PyTorch + CUDA ---------------------------------------")
    try:
        import torch
        v = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"{PASS}PyTorch {v} — CUDA ENABLED")
            print(f"{INFO}GPU: {gpu} ({mem:.1f} GB VRAM)")
        else:
            print(f"{WARN}PyTorch {v} — CPU ONLY (training will be slow!)")
            print(f"{INFO}Install CUDA build:")
            print(f"     pip uninstall torch torchvision torchaudio")
            print(f"     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return True, cuda
    except ImportError:
        print(f"{FAIL}PyTorch not installed in this environment")
        return False, False


def check_scipy():
    """Check scipy (needed by dataset.py)."""
    try:
        import scipy
        print(f"{PASS}scipy {scipy.__version__} installed")
        return True
    except ImportError:
        print(f"{FAIL}scipy not installed (run: pip install scipy)")
        return False


def check_landmarks(landmarks_dir):
    """Check landmarks_300 directory structure and file counts."""
    print("\n--- 2. Landmarks Data ---------------------------------------")
    
    if not os.path.isdir(landmarks_dir):
        print(f"{FAIL}Directory not found: {landmarks_dir}")
        print(f"{INFO}Run landmark extraction first:")
        print(f"     python Model/extract_landmarks.py --workers 4")
        return False

    manifest_path  = os.path.join(landmarks_dir, "manifest.json")
    label_map_path = os.path.join(landmarks_dir, "label_map.json")

    if not os.path.exists(manifest_path):
        print(f"{FAIL}manifest.json not found in {landmarks_dir}")
        return False
    if not os.path.exists(label_map_path):
        print(f"{FAIL}label_map.json not found in {landmarks_dir}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(label_map_path) as f:
        label_map = json.load(f)

    print(f"{PASS}manifest.json and label_map.json found")
    print(f"{INFO}Classes: {len(label_map)}")

    # Count samples per split
    all_ok = True
    for split in ["train", "val", "test"]:
        split_data = manifest.get(split, {})
        n_words = len(split_data)
        n_files = sum(len(v) for v in split_data.values())
        
        # Spot-check first file in each split exists
        missing = 0
        for word, paths in list(split_data.items())[:20]:  # check first 20 words
            for p in paths[:2]:  # check first 2 files per word
                if not os.path.exists(p):
                    missing += 1
        
        if missing > 0:
            print(f"{WARN}[{split}] {n_files} files across {n_words} words — {missing} MISSING (spot check)")
            all_ok = False
        else:
            print(f"{PASS}[{split}] {n_files} samples | {n_words} word classes")

    return all_ok


def check_model():
    """Check model forward pass."""
    print("\n--- 3. Model Forward Pass -----------------------------------")
    try:
        import torch
        from model import GestureTransformer, count_parameters

        # Run 3 architecture
        model = GestureTransformer(
            num_classes=300,
            d_model=192,
            nhead=6,
            num_layers=3,
            dim_ff=384,
            dropout=0.4,
        )
        n = count_parameters(model)
        print(f"{PASS}Model created: {n:,} params ({n/1e6:.2f}M)")

        # Forward pass
        dummy = torch.randn(4, 60, 258)
        out = model(dummy)
        assert out.shape == (4, 300), f"Expected (4, 300), got {out.shape}"
        print(f"{PASS}Forward pass: (4, 60, 258) -> (4, 300) OK")

        # Feature extraction
        logits, feats = model(dummy, return_features=True)
        assert feats.shape == (4, 192), f"Expected (4, 192), got {feats.shape}"
        print(f"{PASS}Feature extraction: (4, 192) OK")

        return True
    except Exception as e:
        print(f"{FAIL}Model check failed: {e}")
        traceback.print_exc()
        return False


def check_dataloader(landmarks_dir):
    """Check DataLoader (one batch from each split)."""
    print("\n--- 4. DataLoader -------------------------------------------")
    try:
        from dataset import get_dataloaders

        train_loader, val_loader, test_loader, num_classes, class_weights = get_dataloaders(
            landmarks_dir, batch_size=4, num_workers=0
        )
        print(f"{PASS}DataLoaders created ({num_classes} classes)")
        print(f"{INFO}Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

        # Load one batch from each split
        for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            batch_x, batch_y = next(iter(loader))
            assert batch_x.shape[1:] == (60, 258), f"{name} batch shape: {batch_x.shape}"
            assert batch_y.dtype in [__import__("torch").int64, __import__("torch").long]
            print(f"{PASS}[{name}] batch shape: {tuple(batch_x.shape)} | labels: {tuple(batch_y.shape)}")

        return True
    except Exception as e:
        print(f"{FAIL}DataLoader check failed: {e}")
        traceback.print_exc()
        return False


def check_mixup():
    """Check mixup functions."""
    print("\n--- 5. Mixup Sanity Check ------------------------------------")
    try:
        import torch
        import torch.nn as nn
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Model"))
        from train import mixup_data, mixup_criterion

        x = torch.randn(8, 60, 258)
        y = torch.randint(0, 300, (8,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
        assert mixed_x.shape == x.shape
        assert 0.0 <= lam <= 1.0

        criterion = nn.CrossEntropyLoss()
        dummy_pred = torch.randn(8, 300)
        loss = mixup_criterion(criterion, dummy_pred, y_a, y_b, lam)
        assert loss.item() > 0

        print(f"{PASS}mixup_data: mixed_x shape {tuple(mixed_x.shape)}, lam={lam:.4f}")
        print(f"{PASS}mixup_criterion: loss = {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"{FAIL}Mixup check failed: {e}")
        traceback.print_exc()
        return False


def check_amp():
    """Check AMP (mixed precision) if CUDA is available."""
    print("\n--- 6. Mixed Precision (AMP) ---------------------------------")
    try:
        import torch
        from torch.amp import GradScaler, autocast

        if not torch.cuda.is_available():
            print(f"{WARN}AMP requires CUDA — skipped (CPU mode)")
            return True

        scaler = GradScaler("cuda", enabled=True)
        with autocast("cuda", enabled=True):
            x = torch.randn(2, 60, 258, device="cuda")
        print(f"{PASS}AMP autocast and GradScaler working on GPU")
        return True
    except Exception as e:
        print(f"{FAIL}AMP check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run 3 pre-flight check")
    parser.add_argument("--landmarks-dir", type=str, default=DEFAULT_LM_DIR)
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  Gestura Run 3 — Pre-flight Check")
    print("=" * 62)
    print(f"  Landmarks dir: {args.landmarks_dir}")
    print()

    results = {}

    torch_ok, cuda_ok = check_pytorch()
    results["pytorch"] = torch_ok

    if not torch_ok:
        print("\n  PyTorch must be installed before continuing.\n")
        sys.exit(1)

    # Check scipy (non-critical, just warn)
    check_scipy()

    results["landmarks"] = check_landmarks(args.landmarks_dir)
    results["model"]     = check_model()

    if results["landmarks"]:
        results["dataloader"] = check_dataloader(args.landmarks_dir)
    else:
        results["dataloader"] = False
        print(f"\n{WARN}Skipping DataLoader check (landmarks not found)")

    results["mixup"] = check_mixup()
    results["amp"]   = check_amp()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Pre-flight Summary")
    print("=" * 62)
    all_ok = True
    for name, ok in results.items():
        icon = "[OK]" if ok else "[!!]"
        status = "PASS" if ok else "FAIL"
        print(f"  {icon} {name:<15} {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  *** ALL CHECKS PASSED — Ready to launch Run 3! ***")
        print()
        print("  Launch command:")
        print("    .\\run3_launch.ps1")
        print("  Or directly:")
        print(f"    python Model\\train.py --epochs 500 --batch-size 32")
    else:
        print("  [!] Some checks FAILED — fix issues above before training")
    print()


if __name__ == "__main__":
    main()
