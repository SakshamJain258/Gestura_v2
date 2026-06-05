"""
startup_checks.py - pre-launch validation helpers.
"""

from pathlib import Path

import cv2


PHASE1_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE1_DIR.parent
MODEL_CANDIDATES = [
    PHASE1_DIR / "best_gesture_model_wlasl300_inference.pt",
    PROJECT_ROOT / "MODEL_Training" / "Model" / "best_gesture_model_wlasl300_inference.pt",
    PHASE1_DIR / "best_gesture_model_wlasl300_epoch471.pt",
    PROJECT_ROOT / "MODEL_Training" / "Model" / "best_gesture_model_wlasl300_epoch471.pt",
]


def check_model_file(paths: list[Path] | None = None) -> tuple[bool, str]:
    """Verify model weights exist before starting workers."""
    paths = paths or MODEL_CANDIDATES
    for path in paths:
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            return True, f"Model found: {path.name} ({size_mb:.1f} MB)"

    expected = "\n".join(f"  - {path}" for path in paths)
    return False, (
        "Model file not found.\n\n"
        f"Expected one of:\n{expected}\n\n"
        "Place your .pt model in Phase_1 or MODEL_Training/Model."
    )


def check_camera(index: int = 0) -> tuple[bool, str]:
    """Verify camera can be opened and read from."""
    cap = cv2.VideoCapture(index)

    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            return True, f"Camera {index} is available."
        return False, (
            f"Camera {index} opened but could not read frames.\n"
            "Try a different camera index."
        )

    cap.release()
    return False, (
        f"Camera {index} could not be opened.\n\n"
        "Possible causes:\n"
        "  - No webcam connected\n"
        "  - Camera in use by another app (close Zoom/Meet/Teams)\n"
        "  - Wrong camera index (try 0 or 1)\n"
    )


def check_pyvirtualcam() -> tuple[bool, str]:
    """Check optional virtual cam dependency."""
    try:
        import pyvirtualcam  # noqa: F401

        return True, "pyvirtualcam is available."
    except ImportError:
        return False, (
            "pyvirtualcam is not installed.\n"
            "Virtual camera output will be disabled.\n\n"
            "To enable: pip install pyvirtualcam"
        )


def run_all_checks(camera_index: int = 0) -> list[dict]:
    """Run all startup checks and return structured results."""
    results = []

    ok, msg = check_model_file()
    results.append({"name": "Model File", "ok": ok, "message": msg, "required": True})

    ok, msg = check_camera(camera_index)
    results.append(
        {
            "name": f"Camera (index {camera_index})",
            "ok": ok,
            "message": msg,
            "required": True,
        }
    )

    ok, msg = check_pyvirtualcam()
    results.append(
        {
            "name": "Virtual Camera (pyvirtualcam)",
            "ok": ok,
            "message": msg,
            "required": False,
        }
    )

    return results


def can_launch(results: list[dict]) -> bool:
    """Return True only if all required checks pass."""
    return all(result["ok"] for result in results if result["required"])
