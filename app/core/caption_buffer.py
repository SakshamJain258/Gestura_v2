"""
CaptionBuffer — centralized sentence state manager.

Replaces the inline _sentence list in InferenceThread with a proper
component that:
  - Tracks the rolling word history
  - Deduplicates consecutive identical words
  - Enforces a max visible word count
  - Tracks time since last word (for LLM pause-trigger in Task 9)
  - Emits a signing_paused signal when no new words arrive for
    PAUSE_THRESHOLD_SEC seconds

Usage:
    buffer = CaptionBuffer(max_words=5, pause_threshold_sec=2.5)
    buffer.signing_paused.connect(my_llm_trigger_slot)
    buffer.append("hello")
    sentence = buffer.get_sentence()   # "hello"
    buffer.clear()
"""

import time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class CaptionBuffer(QObject):
    """Thread-safe word accumulator with pause detection.

    Signals:
        sentence_changed(str): Emitted whenever the visible sentence changes.
        signing_paused(list):  Emitted when no new word arrives for
                               pause_threshold_sec. Payload = word history.
    """

    sentence_changed = pyqtSignal(str)
    signing_paused = pyqtSignal(list)

    def __init__(
        self,
        max_words: int = 5,
        pause_threshold_sec: float = 2.5,
        parent=None,
    ):
        super().__init__(parent)
        self._max_words = max_words
        self._pause_threshold_sec = pause_threshold_sec

        self._sentence: list[str] = []        # visible rolling window
        self._full_history: list[str] = []    # complete session history
        self._last_word_time: float = 0.0
        self._pause_emitted: bool = False      # avoid repeated pause signals

        # Pause detector timer (fires every 500 ms, checks elapsed time)
        self._pause_timer = QTimer(self)
        self._pause_timer.setInterval(500)
        self._pause_timer.timeout.connect(self._check_pause)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Begin pause detection. Call after connecting signals."""
        self._pause_timer.start()

    def stop(self):
        """Stop pause detection and clear state."""
        self._pause_timer.stop()
        self.clear()

    def append(self, word: str):
        """Add a word if it differs from the last word in the sentence."""
        if self._sentence and self._sentence[-1] == word:
            return  # deduplicate consecutive repetitions

        self._sentence.append(word)
        self._full_history.append(word)

        # Keep sentence within max_words rolling window
        if len(self._sentence) > self._max_words:
            self._sentence = self._sentence[-self._max_words:]

        self._last_word_time = time.monotonic()
        self._pause_emitted = False

        self.sentence_changed.emit(self.get_sentence())

    def clear(self):
        """Reset sentence (keep full history for LLM context)."""
        self._sentence.clear()
        self._last_word_time = 0.0
        self._pause_emitted = False
        self.sentence_changed.emit("")

    def full_reset(self):
        """Clear everything including history (new session)."""
        self._sentence.clear()
        self._full_history.clear()
        self._last_word_time = 0.0
        self._pause_emitted = False
        self.sentence_changed.emit("")

    def get_sentence(self) -> str:
        """Return current visible sentence as a space-joined string."""
        return " ".join(self._sentence)

    def get_history(self) -> list[str]:
        """Return full word history for this session."""
        return list(self._full_history)

    def set_max_words(self, n: int):
        self._max_words = max(1, n)

    def set_pause_threshold(self, seconds: float):
        self._pause_threshold_sec = max(0.5, seconds)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_pause(self):
        """Called periodically by QTimer to detect signing pauses."""
        if self._pause_emitted:
            return
        if not self._full_history:
            return
        if self._last_word_time == 0.0:
            return

        elapsed = time.monotonic() - self._last_word_time
        if elapsed >= self._pause_threshold_sec:
            self._pause_emitted = True
            self.signing_paused.emit(list(self._full_history))
