"""
TemporalSmoother

PURPOSE: Prevent the subtitle text from flickering.

THE PROBLEM:
  Models are noisy. Even a good model might output:
    frame 1: "hello"
    frame 2: "yes"      ← a fluke
    frame 3: "hello"
    frame 4: "hello"

  Without smoothing, "yes" flashes on screen for one frame.
  That looks broken and is unreadable.

THE SOLUTION: Voting window
  We keep a history of the last N predictions.
  A prediction is only considered "stable" (real) if it appears
  at least `min_votes` times in that window AND its confidence
  exceeds the threshold.

ANALOGY: It's like asking 10 friends "what did he say?"
  You only believe it if at least 7 of them agree.

EXAMPLE:
  window = 10, min_votes = 7
  history = [hello, hello, hello, yes, hello, hello, hello, hello, hello, hello]
  "hello" appears 9 times → STABLE → output "hello"
  "yes" appears 1 time → UNSTABLE → ignore
"""

from collections import deque, Counter


class TemporalSmoother:
    def __init__(self, window: int = 10, min_votes: int = 7):
        """
        Args:
            window:    How many recent predictions to consider
            min_votes: Minimum times a prediction must appear to be "stable"
        """
        self.window = window
        self.min_votes = min_votes
        self._history = deque(maxlen=window)

    def update(self, prediction_index: int, confidence: float, threshold: float):
        """
        Add a new prediction and return the stable prediction (or None).

        Args:
            prediction_index: The argmax index from model output
            confidence:       The confidence score (0.0 to 1.0)
            threshold:        Minimum confidence to even consider this prediction

        Returns:
            int or None: Stable prediction index, or None if not yet stable
        """
        # Only add to history if confidence meets the threshold
        # Low-confidence predictions count as "noise" — we record them as -1
        # so they dilute any genuine prediction that might be trying to win
        if confidence >= threshold:
            self._history.append(prediction_index)
        else:
            self._history.append(-1)   # -1 = "uncertain"

        # Start reporting as soon as a class has enough votes. Waiting for the
        # full window adds avoidable latency during the first prediction.
        if len(self._history) < self.min_votes:
            return None

        # Count how many times each prediction appears
        counts = Counter(self._history)
        most_common_prediction, vote_count = counts.most_common(1)[0]

        # -1 is not a real prediction
        if most_common_prediction == -1:
            return None

        # Only return if it won enough votes
        if vote_count >= self.min_votes:
            return most_common_prediction

        return None

    def reset(self):
        """Clear history — call this when user resets the sentence."""
        self._history.clear()
