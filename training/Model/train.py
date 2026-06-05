"""
Training Script — WLASL 300 Gesture Transformer (v2, Anti-Overfitting)
=======================================================================
Full training loop with:
  - Mixed precision (AMP) on GPU
  - Cosine annealing LR schedule
  - Class-weighted cross-entropy loss
  - Mixup training (NEW in Run 3 — Fix 2)
  - Label smoothing 0.2 (was 0.05 — Fix 3)
  - Early stopping on val_loss (was val_acc — Fix 4)
  - Reduced model defaults (Fix 5)
  - Top-1 and Top-5 accuracy tracking
  - Best checkpoint saving
  - Training history logging

Changes from Run 2 → Run 3:
  Fix 2: Added mixup_data() and mixup_criterion() for synthetic sample blending
  Fix 3: label_smoothing 0.05 → 0.2 (prevents overconfidence on 8 samples/word)
  Fix 4: Checkpoint on val_loss (smoother signal), patience 50 → 40
  Fix 5: Default model sizes reduced (d_model=192, nhead=6, layers=3, ff=384, dropout=0.4)

Usage:
    python train.py                           # full training
    python train.py --epochs 1 --max-samples 20   # quick test
    python train.py --resume checkpoint.pt     # resume training
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Local imports
from model import GestureTransformer, count_parameters
from dataset import get_dataloaders, WLASLDataset


# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "landmarks_300")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR)
LOG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "training_logs")


# ── Mixup Training (Fix 2) ─────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """
    Mixup: blend two samples to create synthetic training data.
    
    Creates a convex combination of two random samples:
        mixed_x = λ * x_i + (1-λ) * x_j
        loss = λ * L(pred, y_i) + (1-λ) * L(pred, y_j)
    
    This forces the model to learn smooth decision boundaries instead of
    memorizing individual training samples. Proven to significantly reduce
    overfitting on small datasets.
    
    Args:
        x: input batch (B, T, D)
        y: labels (B,)
        alpha: Beta distribution parameter. Higher = more mixing.
               alpha=0.2 means most λ values are near 0 or 1 (mild mixing).
    
    Returns:
        mixed_x: blended input
        y_a, y_b: original labels for both mixed samples
        lam: mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixup-blended samples.
    
    Instead of a single cross-entropy loss, we compute a weighted combination
    of losses against both original labels, proportional to the mixing ratio.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training & Evaluation ──────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, use_mixup=True, mixup_alpha=0.2):
    """Train for one epoch with optional mixup. Returns (loss, top1_acc, top5_acc)."""
    model.train()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            if use_mixup:
                # Mixup: blend two samples for regularization (Fix 2)
                inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
                outputs = model(inputs_mixed)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

        # For accuracy tracking, use the original (non-mixed) predictions
        # When using mixup, we compute accuracy against the primary label (targets_a)
        with torch.no_grad():
            if use_mixup:
                # Re-forward with original inputs for clean accuracy measurement
                with autocast('cuda', enabled=use_amp):
                    clean_outputs = model(inputs)
                acc_outputs = clean_outputs
                acc_labels = labels
            else:
                acc_outputs = outputs
                acc_labels = labels

            # Top-1 accuracy
            _, pred = acc_outputs.topk(1, dim=1)
            correct_top1 += pred.squeeze(1).eq(acc_labels).sum().item()

            # Top-5 accuracy
            _, pred5 = acc_outputs.topk(min(5, acc_outputs.size(1)), dim=1)
            correct_top5 += sum(acc_labels[i].item() in pred5[i].tolist() for i in range(acc_labels.size(0)))

    avg_loss = total_loss / total
    top1_acc = correct_top1 / total * 100
    top5_acc = correct_top5 / total * 100

    return avg_loss, top1_acc, top5_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """Evaluate on val/test set. Returns (loss, top1_acc, top5_acc)."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

        _, pred = outputs.topk(1, dim=1)
        correct_top1 += pred.squeeze(1).eq(labels).sum().item()

        _, pred5 = outputs.topk(min(5, outputs.size(1)), dim=1)
        correct_top5 += sum(labels[i].item() in pred5[i].tolist() for i in range(labels.size(0)))

    avg_loss = total_loss / total
    top1_acc = correct_top1 / total * 100
    top5_acc = correct_top5 / total * 100

    return avg_loss, top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Train GestureTransformer on WLASL-300 (v2 — Anti-Overfitting)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience (on val_loss)")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs")

    # Model architecture (Fix 5 — reduced defaults)
    parser.add_argument("--d-model", type=int, default=192, help="Transformer dim (was 256)")
    parser.add_argument("--nhead", type=int, default=6, help="Attention heads (was 8)")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer layers (was 4)")
    parser.add_argument("--dim-ff", type=int, default=384, help="FFN dimension (was 512)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate (was 0.3)")

    # Mixup (Fix 2)
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup alpha (0 = disabled)")
    parser.add_argument("--no-mixup", action="store_true", help="Disable mixup training")

    # Data
    parser.add_argument("--landmarks-dir", type=str, default=LANDMARKS_DIR)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")

    # Misc
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Training device")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback if CUDA is not available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # ── Setup ───────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        message = (
            "CUDA GPU was requested, but this Python environment has CPU-only PyTorch.\n\n"
            "Install CUDA PyTorch in the active environment:\n\n"
            "  pip uninstall torch torchvision torchaudio\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n\n"
            "Then verify:\n\n"
            "  python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\"\n"
        )
        if not args.allow_cpu:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_amp = device.type == "cuda" and not args.no_amp
    use_mixup = not args.no_mixup and args.mixup_alpha > 0

    print(f"\n{'='*60}")
    print(f"  GestureTransformer Training (v2 — Anti-Overfitting)")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    if device.type == "cuda":
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  CUDA:         {torch.version.cuda}")
    print(f"  AMP:          {use_amp}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch:        {args.batch_size}")
    print(f"  LR:           {args.lr} → {args.min_lr}")
    print(f"  Patience:     {args.patience} (on val_loss)")
    print(f"  Mixup:        {'ON (α={})'.format(args.mixup_alpha) if use_mixup else 'OFF'}")
    print(f"  Label smooth: 0.2")
    print(f"  Model:        d={args.d_model}, heads={args.nhead}, layers={args.num_layers}, ff={args.dim_ff}")
    print(f"  Dropout:      {args.dropout}")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes, class_weights = get_dataloaders(
        args.landmarks_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Classes: {num_classes}")
    print(f"  Train:   {len(train_loader.dataset)} samples")
    print(f"  Val:     {len(val_loader.dataset)} samples")
    print(f"  Test:    {len(test_loader.dataset)} samples")

    # ── Model ───────────────────────────────────────────────────────────────
    model = GestureTransformer(
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Model:   {n_params:,} parameters ({n_params/1e6:.1f}M)\n")

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Fix 3: label_smoothing 0.05 → 0.2
    # With only 8 samples/word, the model becomes overconfident on training
    # examples. 0.2 smoothing distributes probability mass across all classes,
    # preventing the model from assigning 99% probability to memorized samples.
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Smooth cosine annealing over full training (no restarts — stable for small datasets)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
    )

    scaler = GradScaler('cuda', enabled=use_amp)

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top5": [], "lr": []}

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        history = checkpoint.get("history", history)
        print(f"  Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}\n")

    # ── Training Loop ───────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val T5':>6} | {'LR':>10} | {'Time':>6}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Warmup: linear LR ramp during first warmup_epochs
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Train (with mixup if enabled)
        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
            use_mixup=use_mixup, mixup_alpha=args.mixup_alpha,
        )

        # Validate
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device, use_amp)

        # Step scheduler (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}% | {val_top5:>5.1f}% | {current_lr:>10.6f} | {epoch_time:>5.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5"].append(val_top5)
        history["lr"].append(current_lr)

        # ── Checkpointing (Fix 4: on val_loss, not val_acc) ────────────────
        # Val loss is a smoother signal than val accuracy with small datasets.
        # A single sample flipping changes accuracy by ~0.15%, making val_acc
        # noisy. Val loss captures gradual improvement more reliably.
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            best_path = os.path.join(CHECKPOINT_DIR, "gesture_model_300.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_ff": args.dim_ff,
                "history": history,
            }, best_path)
            print(f"       ★ New best model saved! val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        else:
            patience_counter += 1

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "history": history,
            }, periodic_path)

        # ── Early Stopping (Fix 4: based on val_loss) ──────────────────────
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={args.patience}, monitoring val_loss)")
            break

    # ── Final Evaluation ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete — Loading Best Model")
    print(f"{'='*60}")

    best_path = os.path.join(CHECKPOINT_DIR, "gesture_model_300.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Best model from epoch {checkpoint['epoch']+1}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  Best val accuracy: {checkpoint['best_val_acc']:.2f}%")

    # Test evaluation
    test_loss, test_acc, test_top5 = evaluate(model, test_loader, criterion, device, use_amp)
    print(f"\n  Test Results:")
    print(f"    Loss:      {test_loss:.4f}")
    print(f"    Top-1 Acc: {test_acc:.2f}%")
    print(f"    Top-5 Acc: {test_top5:.2f}%")

    # ── Save Training History ───────────────────────────────────────────────
    history_path = os.path.join(LOG_DIR, "training_history.json")
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["test_top5"] = test_top5
    history["best_val_acc"] = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["total_epochs"] = epoch + 1
    history["num_classes"] = num_classes
    history["num_params"] = count_parameters(model)
    history["fixes_applied"] = [
        "Fix 1: Stronger augmentation (noise 0.03, dropout 25%, speed 0.70-1.30, +hand_scaling, +spatial_jitter)",
        "Fix 2: Mixup training (alpha=0.2)",
        "Fix 3: Label smoothing 0.2 (was 0.05)",
        "Fix 4: Early stopping on val_loss (was val_acc), patience=40",
        "Fix 5: Reduced model (d=192, heads=6, layers=3, ff=384, dropout=0.4)",
    ]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved → {history_path}")

    # ── Export inference-ready model ────────────────────────────────────────
    inference_path = os.path.join(CHECKPOINT_DIR, "gesture_model_300_inference.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_ff": args.dim_ff,
        "dropout": 0.0,  # No dropout at inference
    }, inference_path)
    print(f"  Inference model saved → {inference_path}")

    print(f"\n{'='*60}")
    print(f"  Done! ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
