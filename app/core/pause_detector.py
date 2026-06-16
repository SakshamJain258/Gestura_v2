"""
PauseDetector — triggers the Gemini LLM when the signer pauses.

Connects to CaptionBuffer.signing_paused and GeminiClient.refine_caption.

The detector bridges between:
  - CaptionBuffer → detects pause after word accumulation
  - GeminiClient  → calls the LLM for refinement

Usage:
    detector = PauseDetector(caption_buffer, gemini_client)
    detector.refined_caption.connect(ui_llm_label.setText)
    detector.start()
"""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class PauseDetector(QObject):
    """
    Bridges CaptionBuffer.signing_paused → GeminiClient.refine_caption.

    Signals:
        refined_caption(str): Re-emits GeminiClient.caption_refined
        llm_active(bool):     True while a Gemini request is in progress
    """

    refined_caption = pyqtSignal(str)
    llm_active = pyqtSignal(bool)

    def __init__(self, caption_buffer=None, gemini_client=None, parent=None):
        super().__init__(parent)
        self._caption_buffer = caption_buffer
        self._gemini_client = gemini_client
        self._enabled = True
        self._min_words = 1    # need at least this many words to trigger Gemini

    # ── Setup ─────────────────────────────────────────────────────────────────

    def connect_all(self):
        """Wire CaptionBuffer and GeminiClient signals."""
        if self._caption_buffer is not None:
            self._caption_buffer.signing_paused.connect(self._on_signing_paused)

        if self._gemini_client is not None:
            self._gemini_client.caption_refined.connect(self.refined_caption)
            self._gemini_client.status_changed.connect(self._on_gemini_status)

    def set_enabled(self, enabled: bool):
        """Enable or disable LLM triggering (e.g., no API key configured)."""
        self._enabled = enabled
        if self._gemini_client is not None:
            self._gemini_client.set_enabled(enabled)

    def set_min_words(self, n: int):
        """Minimum accumulated words before triggering LLM refinement."""
        self._min_words = max(1, n)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot(list)
    def _on_signing_paused(self, word_history: list):
        """Triggered by CaptionBuffer when signing stops."""
        if not self._enabled:
            return

        # Filter out noise / partial words
        meaningful_words = [w for w in word_history if w.strip()]
        if len(meaningful_words) < self._min_words:
            return

        if self._gemini_client is not None:
            self.llm_active.emit(True)
            self._gemini_client.refine_caption(meaningful_words)

    @pyqtSlot(str)
    def _on_gemini_status(self, status: str):
        is_working = "refin" in status.lower()
        self.llm_active.emit(is_working)
