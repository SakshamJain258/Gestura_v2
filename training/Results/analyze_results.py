"""
Training Results Analyzer - Gestura WLASL 300
==============================================
Generates a comprehensive analysis of training runs:
  - Training curves (loss, accuracy, LR)
  - Summary statistics
  - Per-run comparison

Usage:
    python analyze_results.py                    # analyze latest run
    python analyze_results.py --run-dir Run_2    # analyze specific run
"""

import json
import os
import sys
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_history(history_path):
    with open(history_path) as f:
        return json.load(f)


def print_summary(h, run_name="Training Run"):
    print("\n" + "="*60)
    print(f"  {run_name} - Summary")
    print("="*60)
    
    total = h.get('total_epochs', len(h['train_loss']))
    best_val_idx = max(range(len(h['val_acc'])), key=lambda i: h['val_acc'][i])
    
    print("\n  [TRAINING OVERVIEW]")
    print("  " + "-"*40)
    print(f"  Total epochs:        {total}")
    print(f"  Parameters:          {h.get('num_params', 'N/A'):,}")
    print(f"  Classes:             {h.get('num_classes', 300)}")
    
    print("\n  [TRAINING METRICS - Final Epoch]")
    print("  " + "-"*40)
    print(f"  Train loss:          {h['train_loss'][-1]:.4f}")
    print(f"  Train accuracy:      {h['train_acc'][-1]:.2f}%")
    print(f"  Peak train acc:      {max(h['train_acc']):.2f}%")
    
    print("\n  [VALIDATION METRICS]")
    print("  " + "-"*40)
    print(f"  Best val accuracy:   {h.get('best_val_acc', max(h['val_acc'])):.2f}%")
    print(f"  Best epoch:          {best_val_idx + 1}")
    print(f"  Final val accuracy:  {h['val_acc'][-1]:.2f}%")
    print(f"  Final val top-5:     {h['val_top5'][-1]:.2f}%")
    
    if 'test_acc' in h:
        print("\n  [TEST RESULTS - Best Model]")
        print("  " + "-"*40)
        print(f"  Test loss:           {h['test_loss']:.4f}")
        print(f"  Test top-1 acc:      {h['test_acc']:.2f}%")
        print(f"  Test top-5 acc:      {h['test_top5']:.2f}%")
    
    gap = h['train_acc'][-1] - h['val_acc'][-1]
    print("\n  [ANALYSIS]")
    print("  " + "-"*40)
    print(f"  Train-Val gap:       {gap:.2f}% (final epoch)")
    if gap > 20:
        print(f"  !! Significant overfitting detected ({gap:.0f}% gap)")
        print(f"     Consider: more data augmentation, higher dropout, or fewer epochs")
    elif gap > 10:
        print(f"  >> Moderate overfitting ({gap:.0f}% gap)")
    else:
        print(f"  OK Good generalization ({gap:.0f}% gap)")
    
    print("\n" + "="*60 + "\n")


def plot_training_curves(h, save_dir, run_name="Training Run"):
    if not HAS_MATPLOTLIB:
        print("  [!] matplotlib not installed -- skipping plots")
        print("      Install with: pip install matplotlib")
        return None
    
    epochs = range(1, len(h['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{run_name} - Training Curves', fontsize=14, fontweight='bold')
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, h['train_loss'], label='Train Loss', color='#2196F3', alpha=0.8)
    ax.plot(epochs, h['val_loss'], label='Val Loss', color='#F44336', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, h['train_acc'], label='Train Acc', color='#2196F3', alpha=0.8)
    ax.plot(epochs, h['val_acc'], label='Val Acc', color='#F44336', alpha=0.8)
    ax.plot(epochs, h['val_top5'], label='Val Top-5', color='#4CAF50', alpha=0.8, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, h['lr'], color='#FF9800', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Overfitting gap
    ax = axes[1, 1]
    gap = [t - v for t, v in zip(h['train_acc'], h['val_acc'])]
    ax.plot(epochs, gap, color='#9C27B0', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(epochs, gap, alpha=0.2, color='#9C27B0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Acc - Val Acc (%)')
    ax.set_title('Overfitting Gap')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Training curves saved -> {plot_path}")
    
    return plot_path


def generate_report(h, save_dir, run_name="Training Run"):
    total = h.get('total_epochs', len(h['train_loss']))
    best_val_idx = max(range(len(h['val_acc'])), key=lambda i: h['val_acc'][i])
    gap = h['train_acc'][-1] - h['val_acc'][-1]
    
    report = f"""# {run_name} - Training Report

**Generated:** Auto-generated by analyze_results.py  
**Model:** GestureTransformer ({h.get('num_params', 'N/A'):,} parameters)  
**Classes:** {h.get('num_classes', 300)}  
**Total Epochs:** {total}

---

## Results Summary

| Metric | Value |
|---|---|
| **Best Val Accuracy** | {h.get('best_val_acc', max(h['val_acc'])):.2f}% |
| **Best Epoch** | {best_val_idx + 1} |
| **Test Top-1 Accuracy** | {h.get('test_acc', 'N/A'):.2f}% |
| **Test Top-5 Accuracy** | {h.get('test_top5', 'N/A'):.2f}% |
| **Test Loss** | {h.get('test_loss', 'N/A'):.4f} |
| **Peak Train Accuracy** | {max(h['train_acc']):.2f}% |
| **Final Train Loss** | {h['train_loss'][-1]:.4f} |
| **Train-Val Gap** | {gap:.2f}% |

## Training Progression

| Epoch | Train Loss | Train Acc | Val Acc | Val Top-5 |
|---|---|---|---|---|
"""
    
    milestones = [1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for e in milestones:
        if e <= total:
            i = e - 1
            report += f"| {e} | {h['train_loss'][i]:.4f} | {h['train_acc'][i]:.2f}% | {h['val_acc'][i]:.2f}% | {h['val_top5'][i]:.2f}% |\n"
    
    report += f"| **{best_val_idx+1} (best)** | **{h['train_loss'][best_val_idx]:.4f}** | **{h['train_acc'][best_val_idx]:.2f}%** | **{h['val_acc'][best_val_idx]:.2f}%** | **{h['val_top5'][best_val_idx]:.2f}%** |\n"
    
    report += f"""
## Analysis

- **Overfitting:** {"!! Significant" if gap > 20 else ">> Moderate" if gap > 10 else "OK Good"} (train-val gap = {gap:.1f}%)
- **Convergence:** {"Model converged" if h['train_loss'][-1] < 2.0 else "May need more epochs"}
- **Learning Rate:** Started at {h['lr'][0]:.6f}, ended at {h['lr'][-1]:.8f}

## Training Curves

![Training Curves](training_curves.png)
"""
    
    report_path = os.path.join(save_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  [+] Training report saved -> {report_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze Gestura training results")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory to analyze")
    parser.add_argument("--history", type=str, default=None, help="Direct path to training_history.json")
    args = parser.parse_args()
    
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.history:
        history_path = args.history
        save_dir = os.path.dirname(history_path)
    elif args.run_dir:
        run_path = os.path.join(results_dir, args.run_dir)
        history_path = os.path.join(run_path, 'training_history.json')
        save_dir = run_path
    else:
        runs = sorted([d for d in os.listdir(results_dir) if d.startswith('Run_') and os.path.isdir(os.path.join(results_dir, d))])
        if runs:
            save_dir = os.path.join(results_dir, runs[-1])
            history_path = os.path.join(save_dir, 'training_history.json')
        else:
            history_path = os.path.join(os.path.dirname(results_dir), 'training_logs', 'training_history.json')
            save_dir = os.path.dirname(history_path)
    
    if not os.path.exists(history_path):
        print(f"  [X] No training history found at: {history_path}")
        sys.exit(1)
    
    run_name = os.path.basename(save_dir)
    h = load_history(history_path)
    
    print_summary(h, run_name)
    plot_training_curves(h, save_dir, run_name)
    generate_report(h, save_dir, run_name)
    
    print(f"\n  [OK] All analysis files saved to: {save_dir}")


if __name__ == "__main__":
    main()
