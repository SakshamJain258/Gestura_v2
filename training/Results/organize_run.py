"""
Organize Training Results — Move files to a structured Run directory.

Usage:
    python organize_run.py --run-name Run_2 --description "500 epochs, fixed LR schedule"
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Organize training run results")
    parser.add_argument("--run-name", type=str, required=True, help="Name for this run (e.g., Run_2)")
    parser.add_argument("--description", type=str, default="", help="Description of this run")
    parser.add_argument("--keep-checkpoints", nargs="*", type=int, default=None,
                        help="Which epoch checkpoints to keep (e.g., 100 200 500). Default: best + last")
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(script_dir), "Model")
    logs_dir = os.path.join(os.path.dirname(script_dir), "training_logs")
    run_dir = os.path.join(script_dir, args.run_name)
    
    # Create run directory structure
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Organizing → {args.run_name}")
    print(f"{'='*60}\n")
    
    moved = []
    
    # 1. Copy best model
    best_model = os.path.join(model_dir, "gesture_model_300.pt")
    if os.path.exists(best_model):
        dest = os.path.join(run_dir, "gesture_model_300_best.pt")
        shutil.copy2(best_model, dest)
        print(f"  ✅ Best model → {os.path.basename(dest)}")
        moved.append(("best_model", dest))
    
    # 2. Copy inference model
    inf_model = os.path.join(model_dir, "gesture_model_300_inference.pt")
    if os.path.exists(inf_model):
        dest = os.path.join(run_dir, "gesture_model_300_inference.pt")
        shutil.copy2(inf_model, dest)
        print(f"  ✅ Inference model → {os.path.basename(dest)}")
        moved.append(("inference_model", dest))
    
    # 3. Copy training history
    history_src = os.path.join(logs_dir, "training_history.json")
    if os.path.exists(history_src):
        dest = os.path.join(run_dir, "training_history.json")
        shutil.copy2(history_src, dest)
        print(f"  ✅ Training history → {os.path.basename(dest)}")
        moved.append(("training_history", dest))
    
    # 4. Move periodic checkpoints
    checkpoint_count = 0
    for f in sorted(os.listdir(model_dir)):
        if f.startswith("checkpoint_epoch_") and f.endswith(".pt"):
            epoch_num = int(f.replace("checkpoint_epoch_", "").replace(".pt", ""))
            
            # Keep only specified checkpoints (or all if not specified)
            if args.keep_checkpoints is not None and epoch_num not in args.keep_checkpoints:
                # Move to checkpoints subfolder anyway (just organized)
                src = os.path.join(model_dir, f)
                dest = os.path.join(run_dir, "checkpoints", f)
                shutil.move(src, dest)
            else:
                src = os.path.join(model_dir, f)
                dest = os.path.join(run_dir, "checkpoints", f)
                shutil.move(src, dest)
            checkpoint_count += 1
    
    if checkpoint_count > 0:
        print(f"  ✅ {checkpoint_count} checkpoints → checkpoints/")
    
    # 5. Create run metadata
    metadata = {
        "run_name": args.run_name,
        "description": args.description,
        "timestamp": datetime.now().isoformat(),
        "files": {k: os.path.basename(v) for k, v in moved},
        "checkpoints_count": checkpoint_count,
    }
    
    # Add training results to metadata
    if os.path.exists(os.path.join(run_dir, "training_history.json")):
        with open(os.path.join(run_dir, "training_history.json")) as f:
            h = json.load(f)
            metadata["results"] = {
                "total_epochs": h.get("total_epochs", len(h["train_loss"])),
                "best_val_acc": h.get("best_val_acc", max(h["val_acc"])),
                "test_top1": h.get("test_acc", None),
                "test_top5": h.get("test_top5", None),
                "num_params": h.get("num_params", None),
                "num_classes": h.get("num_classes", None),
            }
    
    meta_path = os.path.join(run_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata → run_metadata.json")
    
    print(f"\n  📁 Results organized in: {run_dir}")
    print(f"\n  Run analysis with:")
    print(f"    python analyze_results.py --run-dir {args.run_name}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
