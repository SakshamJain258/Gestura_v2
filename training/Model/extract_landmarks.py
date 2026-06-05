"""
WLASL Landmark Extraction Pipeline
===================================
Extracts MediaPipe Holistic landmarks from raw MP4 videos
and saves them as .npy arrays for training.

Output per video: (60, 258) numpy array
  - 60 frames (padded/truncated)
  - 258 features = pose(33×4) + left_hand(21×3) + right_hand(21×3)

Usage:
    python extract_landmarks.py                         # all splits
    python extract_landmarks.py --split train           # one split
    python extract_landmarks.py --split train --workers 4 --max-videos 10
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import mediapipe as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SEQ_LENGTH = 60          # frames per sequence (pad/truncate to this)
POSE_DIMS = 33 * 4       # x, y, z, visibility
HAND_DIMS = 21 * 3       # x, y, z per hand
FEATURE_DIM = POSE_DIMS + HAND_DIMS * 2  # 258

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "WLASL_300")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "landmarks_300")


def extract_keypoints(results):
    """Extract 258-dim feature vector from MediaPipe Holistic results."""
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(POSE_DIMS)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(HAND_DIMS)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(HAND_DIMS)
    )
    return np.concatenate([pose, lh, rh])


def process_video(video_path, seq_length=SEQ_LENGTH):
    """
    Extract landmarks from a single video file.
    Returns: numpy array of shape (seq_length, 258) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_landmarks = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            keypoints = extract_keypoints(results)
            frames_landmarks.append(keypoints)

    cap.release()

    if len(frames_landmarks) == 0:
        return None

    landmarks = np.array(frames_landmarks)

    # Pad or truncate to seq_length
    if len(landmarks) < seq_length:
        pad = np.zeros((seq_length - len(landmarks), FEATURE_DIM))
        landmarks = np.concatenate([landmarks, pad], axis=0)
    else:
        landmarks = landmarks[:seq_length]

    return landmarks  # (60, 258)


def process_single_video(args):
    """Worker function for multiprocessing."""
    video_path, output_path = args

    # Skip if already extracted
    if os.path.exists(output_path):
        return output_path, True, "skipped"

    landmarks = process_video(video_path)
    if landmarks is None:
        return output_path, False, "failed"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, landmarks)
    return output_path, True, "extracted"


def get_video_list(split):
    """Get all (video_path, output_path) pairs for a split."""
    split_dir = os.path.join(DATASET_ROOT, split)
    if not os.path.exists(split_dir):
        print(f"Split directory not found: {split_dir}")
        return []

    pairs = []
    for word_dir in sorted(os.listdir(split_dir)):
        word_path = os.path.join(split_dir, word_dir)
        if not os.path.isdir(word_path):
            continue

        for video_file in sorted(os.listdir(word_path)):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(word_path, video_file)
            npy_name = video_file.replace(".mp4", ".npy")
            output_path = os.path.join(OUTPUT_DIR, split, word_dir, npy_name)
            pairs.append((video_path, output_path))

    return pairs


def build_manifest(splits=("train", "val", "test")):
    """
    Build manifest.json mapping word → list of .npy paths for each split.
    Also builds label_map.json (word → integer index).
    """
    manifest = {}
    all_words = set()

    for split in splits:
        split_dir = os.path.join(OUTPUT_DIR, split)
        if not os.path.exists(split_dir):
            continue

        manifest[split] = {}
        for word_dir in sorted(os.listdir(split_dir)):
            word_path = os.path.join(split_dir, word_dir)
            if not os.path.isdir(word_path):
                continue

            npy_files = sorted([
                os.path.join(word_path, f)
                for f in os.listdir(word_path)
                if f.endswith(".npy")
            ])

            if npy_files:
                manifest[split][word_dir] = npy_files
                all_words.add(word_dir)

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")

    # Build and save label map (sorted alphabetically for consistency)
    sorted_words = sorted(all_words)
    label_map = {word: idx for idx, word in enumerate(sorted_words)}
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved → {label_map_path} ({len(label_map)} classes)")

    return manifest, label_map


def main():
    parser = argparse.ArgumentParser(description="Extract WLASL landmarks")
    parser.add_argument("--split", type=str, default=None,
                        help="Process only this split (train/val/test). Default: all.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2),
                        help="Number of parallel workers.")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max videos to process (for testing).")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in splits:
        pairs = get_video_list(split)
        if not pairs:
            print(f"No videos found for split: {split}")
            continue

        if args.max_videos:
            pairs = pairs[:args.max_videos]

        print(f"\n{'='*60}")
        print(f"  Processing {split}: {len(pairs)} videos | {args.workers} workers")
        print(f"{'='*60}")

        success, failed, skipped = 0, 0, 0

        if args.workers <= 1:
            # Single-process (easier to debug)
            for i, pair in enumerate(pairs):
                path, ok, status = process_single_video(pair)
                if status == "skipped":
                    skipped += 1
                elif ok:
                    success += 1
                else:
                    failed += 1

                if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                    print(f"  [{split}] {i+1}/{len(pairs)} | ✓ {success} | ✗ {failed} | ⊘ {skipped}")
        else:
            # Multi-process
            with Pool(args.workers) as pool:
                for i, (path, ok, status) in enumerate(pool.imap_unordered(process_single_video, pairs)):
                    if status == "skipped":
                        skipped += 1
                    elif ok:
                        success += 1
                    else:
                        failed += 1

                    if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                        print(f"  [{split}] {i+1}/{len(pairs)} | ✓ {success} | ✗ {failed} | ⊘ {skipped}")

        print(f"\n  {split} done: ✓ {success} extracted | ✗ {failed} failed | ⊘ {skipped} skipped")

    # Build manifest after all extraction
    print(f"\n{'='*60}")
    print("  Building manifest...")
    print(f"{'='*60}")
    build_manifest(splits)
    print("\nDone!")


if __name__ == "__main__":
    main()
