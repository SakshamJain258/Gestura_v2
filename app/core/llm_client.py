"""
GeminiClient — async Gemini API integration for caption refinement.

Reads the API key from ~/.gestura/config.json.
Provides refine_caption() which takes a list of raw predicted words
and returns a grammatically polished sentence.

Task 8: Core LLM integration
Task 9: Called by PauseDetector on signing pause
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


# Config file location
CONFIG_DIR = Path.home() / ".gestura"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Prefer new SDK if available; fall back to legacy
try:
    import google.genai as genai_new
    _USE_NEW_SDK = True
    GEMINI_MODEL = "gemini-2.0-flash-lite"
except ImportError:
    _USE_NEW_SDK = False
    GEMINI_MODEL = "gemini-1.5-flash"

# System prompt for caption refinement
SYSTEM_PROMPT = (
    "You are a real-time ASL (American Sign Language) caption assistant. "
    "The user is signing in ASL, and an AI model has recognized these words: {words}. "
    "Your task: produce a single, natural English sentence (or short phrase) that best "
    "represents what the signer likely meant. Fix grammar and word order. "
    "If the word list is short (1-3 words), return it as-is without embellishment. "
    "Return ONLY the refined sentence, nothing else."
)


def load_api_key() -> Optional[str]:
    """Load Gemini API key from ~/.gestura/config.json."""
    if not CONFIG_FILE.exists():
        return None
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            config = json.load(f)
        key = config.get("gemini_api_key", "").strip()
        return key if key else None
    except (json.JSONDecodeError, OSError):
        return None


def save_api_key(api_key: str):
    """Save Gemini API key to ~/.gestura/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {}
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}
    config["gemini_api_key"] = api_key.strip()
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def test_api_key(api_key: str) -> tuple[bool, str]:
    """Synchronously test an API key with a minimal request."""
    try:
        if _USE_NEW_SDK:
            from google import genai as _genai
            client = _genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Say 'ok'",
            )
            return True, f"API key is valid. (SDK: google-genai, model: {GEMINI_MODEL})"
        else:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel(GEMINI_MODEL)
            model.generate_content("Say 'ok'")
            return True, f"API key is valid. (SDK: google-generativeai, model: {GEMINI_MODEL})"
    except Exception as exc:
        err_str = str(exc)
        if "429" in err_str or "Quota" in err_str or "ResourceExhausted" in err_str:
            friendly_msg = (
                "API key test failed: Quota Exceeded (Error 429).\n"
                "Note: A newly created API key can take up to 2-3 minutes to propagate. "
                "Please wait a moment and test again."
            )
            return False, friendly_msg
        return False, f"API key test failed: {exc}"


class GeminiWorker(QThread):
    """Background worker that makes a single Gemini API call."""

    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, words: list[str], api_key: str, parent=None):
        super().__init__(parent)
        self._words = words
        self._api_key = api_key

    def run(self):
        try:
            prompt = SYSTEM_PROMPT.format(words=" | ".join(self._words))

            if _USE_NEW_SDK:
                from google import genai as _genai
                client = _genai.Client(api_key=self._api_key)
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                text = response.text.strip()
            else:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self._api_key)
                model = genai_legacy.GenerativeModel(GEMINI_MODEL)
                response = model.generate_content(prompt)
                text = response.text.strip()

            self.result_ready.emit(text)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "Quota" in err_str or "ResourceExhausted" in err_str:
                friendly_msg = (
                    "Gemini API Quota Exceeded (Error 429).\n"
                    "Note: Newly created API keys can take up to 2-3 minutes to activate. "
                    "If you just created this key, please wait a moment and try again."
                )
                self.error_occurred.emit(friendly_msg)
            else:
                self.error_occurred.emit(f"Gemini error: {exc}")


class GeminiClient(QObject):
    """
    High-level Gemini caption refinement client.

    Fires off a background GeminiWorker thread for each refinement request.
    Signals:
        caption_refined(str): Emits the LLM-polished caption.
        error_occurred(str):  Emits on API errors.
        status_changed(str):  Status messages ("Refining...", "Ready", etc.)
    """

    caption_refined = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[GeminiWorker] = None
        self._enabled = False

    def is_available(self) -> bool:
        """True if at least one Gemini SDK is installed and an API key is configured."""
        sdk_available = _USE_NEW_SDK
        if not sdk_available:
            try:
                import google.generativeai  # noqa: F401
                sdk_available = True
            except ImportError:
                pass
        return sdk_available and load_api_key() is not None

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    @pyqtSlot(list)
    def refine_caption(self, words: list[str]):
        """Trigger async refinement. Ignores if no API key or already running."""
        if not self._enabled:
            return

        api_key = load_api_key()
        if not api_key:
            self.error_occurred.emit("Gemini API key not configured. Open Settings to add it.")
            return

        if not words:
            return

        # Cancel any in-progress request
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self.status_changed.emit("Refining with Gemini...")
        self._worker = GeminiWorker(words, api_key, parent=self)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, text: str):
        self.status_changed.emit("Gemini ready")
        self.caption_refined.emit(text)

    def _on_error(self, message: str):
        self.status_changed.emit("Gemini error")
        self.error_occurred.emit(message)
