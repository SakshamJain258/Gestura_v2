"""
startup_checks.py - pre-launch validation helpers.
"""

import sys
from pathlib import Path


def _get_base_dir() -> Path:
    """Base resource directory — supports both frozen exe and normal Python."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]   # Gestura v2/


_BASE = _get_base_dir()
TRAINING_RESULTS = _BASE / "training" / "Results"

MODEL_CANDIDATES = [
    # Run 3 (current best)
    TRAINING_RESULTS / "Run_3" / "gesture_model_300_inference.pt",
    TRAINING_RESULTS / "Run_3" / "gesture_model_300_best.pt",
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
    """
    Verify camera index is non-negative.
    Actual camera opening is deferred to the background CaptureThread
    to prevent blocking or freezing the main GUI thread.
    """
    if index >= 0:
        return True, f"Camera index {index} is valid."
    return False, f"Invalid camera index {index}."


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
