"""
generate_synthetic_landmarks.py
================================
Generates distinct synthetic right-hand landmark profiles for ASL A-Z + space
to train a functional placeholder FingerspellingClassifier model.
"""

import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LANDMARKS_DIR = SCRIPT_DIR / "fingerspelling_landmarks"

CLASSES = [chr(i) for i in range(65, 91)] + ["space"]  # A-Z + space
SAMPLES_PER_CLASS = 150

def main():
    print(f"Generating synthetic landmarks in: {LANDMARKS_DIR}")
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # Base shapes for hand joints (21 joints, 3 coordinates)
    # We will perturb these systematically per class so they are highly distinct
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = LANDMARKS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating {class_name}...")
        for sample_idx in range(SAMPLES_PER_CLASS):
            # Define a systematic coordinate pattern depending on the class_idx
            # This makes it easy for the neural net to classify, but has realistic limits [0, 1]
            landmarks = np.zeros((21, 3), dtype=np.float32)
            
            # Create a class-specific base shape
            for joint in range(21):
                # Spread out x coordinates
                landmarks[joint, 0] = 0.5 + 0.3 * np.sin(joint * 0.3 + class_idx * 0.2)
                # Spread out y coordinates
                landmarks[joint, 1] = 0.5 + 0.3 * np.cos(joint * 0.3 + class_idx * 0.1)
                # Spread out z coordinates
                landmarks[joint, 2] = -0.1 * (joint / 21.0) + 0.05 * np.cos(class_idx)

            # Add noise
            noise = np.random.normal(0, 0.015, landmarks.shape).astype(np.float32)
            landmarks += noise

            # Flatten to 63 floats
            flat_landmarks = landmarks.flatten()

            out_path = class_dir / f"synth_{sample_idx:04d}.npy"
            np.save(str(out_path), flat_landmarks)

    print("Generation complete!")

if __name__ == "__main__":
    main()
