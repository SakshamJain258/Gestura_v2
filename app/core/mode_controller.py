"""
ModeController — manages Layer 1 / Layer 2 switching for Gestura.

Layer 1: Word-level ASL recognition (GestureTransformer, Run 3)
Layer 2: Fingerspelling (FingerspellingClassifier, letter-by-letter)

The controller owns the current mode state and propagates changes to
InferenceThread and the UI simultaneously.

Usage:
    controller = ModeController(inference_thread)
    controller.mode_changed.connect(ui_label.setText)
    controller.set_mode("finger")
"""

from PyQt6.QtCore import QObject, pyqtSignal


MODE_WORD = "word"
MODE_FINGER = "finger"

MODE_LABELS = {
    MODE_WORD: "Word-level ASL",
    MODE_FINGER: "Fingerspelling A–Z",
}


class ModeController(QObject):
    """Manages recognition mode between word-level ASL and fingerspelling.

    Signals:
        mode_changed(str): Emits the human-readable mode label on each switch.
    """

    mode_changed = pyqtSignal(str)

    def __init__(self, inference_thread=None, parent=None):
        super().__init__(parent)
        self._current_mode = MODE_WORD
        self._inference_thread = inference_thread

    def set_inference_thread(self, thread):
        """Bind to a (re)started InferenceThread."""
        self._inference_thread = thread

    def get_mode(self) -> str:
        return self._current_mode

    def get_mode_label(self) -> str:
        return MODE_LABELS.get(self._current_mode, self._current_mode)

    def set_mode(self, mode: str):
        """Switch to the given mode and propagate to the inference worker."""
        if mode not in MODE_LABELS:
            raise ValueError(f"Unknown mode: {mode!r}")

        if mode == self._current_mode:
            return

        self._current_mode = mode
        if self._inference_thread is not None and self._inference_thread.isRunning():
            self._inference_thread.set_mode(mode)

        self.mode_changed.emit(MODE_LABELS[mode])

    def toggle(self):
        """Cycle between word and finger modes."""
        next_mode = MODE_FINGER if self._current_mode == MODE_WORD else MODE_WORD
        self.set_mode(next_mode)

    def sync_to_thread(self):
        """Re-apply current mode to a freshly started InferenceThread."""
        if self._inference_thread is not None and self._inference_thread.isRunning():
            self._inference_thread.set_mode(self._current_mode)
