# 07 — LLM Integration (Gemini Caption Refinement)

## Overview

The third recognition layer in Gestura uses the **Gemini API** to convert raw sign word streams into grammatically natural English sentences. This is entirely optional — the app works fully offline without it.

---

## Why an LLM Layer?

ASL grammar differs from English grammar. ASL uses a topic-comment structure and often omits articles, auxiliary verbs, and verb inflections. A recognition system that outputs raw word sequences produces output like:

```
you name what
```

An LLM can convert this to:

```
What is your name?
```

This is especially useful when Gestura is being used in a video call where captions are read by hearing participants who expect standard English.

---

## Architecture

```
InferenceThread
    └── word_detected(word) signal
            │
            ▼
       CaptionBuffer
            ├── Appends word to rolling buffer (max 5 words)
            └── Resets signing pause timer (2.5 seconds)
                    │
                    ▼ (after 2.5 seconds of no new words)
               PauseDetector
                    └── Calls GeminiClient.refine_caption(word_list)
                                │
                                ▼
                           GeminiWorker (background QThread)
                                ├── Calls Gemini API (non-blocking)
                                └── Emits result_ready(refined_text)
                                        │
                                        ▼
                               MainWindow._on_llm_caption()
                                        └── Updates llm_label strip in UI
```

---

## API Key Storage

The Gemini API key is stored locally in a plain JSON file:

```
~/.gestura/config.json
```

```json
{
  "gemini_api_key": "AIza..."
}
```

The file is created on first save and never sent anywhere. The key is loaded fresh on each API call (`load_api_key()` in `core/llm_client.py`).

**Security note:** This is a user-local config file, not a repository file. It is not committed to Git (the `.gitignore` entry for `~/.gestura/` prevents accidental inclusion).

---

## SDK Support

The code supports both the new and legacy Gemini Python SDKs:

```python
try:
    import google.genai as genai_new
    _USE_NEW_SDK = True
    GEMINI_MODEL = "gemini-2.0-flash-lite"
except ImportError:
    _USE_NEW_SDK = False
    GEMINI_MODEL = "gemini-1.5-flash"
```

The new `google-genai` SDK (installed via `pip install google-genai`) is preferred because it supports newer model versions. The legacy `google-generativeai` SDK is used as a fallback.

---

## System Prompt

```
You are a real-time ASL (American Sign Language) caption assistant.
The user is signing in ASL, and an AI model has recognized these words: {words}.
Your task: produce a single, natural English sentence (or short phrase) that best
represents what the signer likely meant. Fix grammar and word order.
If the word list is short (1–3 words), return it as-is without embellishment.
Return ONLY the refined sentence, nothing else.
```

Words are passed as a pipe-separated list: `hello | you | name | what`.

---

## Error Handling

| Error | Handling |
|---|---|
| API key not configured | `error_occurred` signal → status bar message |
| Quota exceeded (429) | Friendly message: "Newly created keys take 2–3 min to activate" |
| Any other exception | Logged and emitted as `error_occurred` |

---

## GeminiWorker (Background Thread)

Each refinement request spawns a new `GeminiWorker` (QThread). If a previous request is still running when a new one is triggered, the old worker is terminated:

```python
if self._worker and self._worker.isRunning():
    self._worker.terminate()
    self._worker.wait()
```

This prevents stale responses from a slow API call overwriting a newer response.

---

## PauseDetector

`core/pause_detector.py` watches the CaptionBuffer and decides when to trigger Gemini refinement:

```python
class PauseDetector:
    def __init__(self, buffer: CaptionBuffer, gemini: GeminiClient):
        ...
    def connect_all(self):
        self._buffer.signing_paused.connect(self._on_pause)
    
    def _on_pause(self, words: list[str]):
        if self._enabled and words:
            self._gemini.refine_caption(words)
```

The CaptionBuffer emits `signing_paused` after 2.5 seconds of no new word detections. This ensures Gemini is only called when the user appears to have finished a phrase — not after every word.

---

## UI Controls

| Control | Location | Behavior |
|---|---|---|
| **⚙ Key** button | Right panel | Opens `ApiKeyDialog` to enter/test/save key |
| **Enable Gemini** toggle | Right panel | Activates/deactivates the LLM layer |
| **Gemini status label** | Right panel | Shows "API key configured ✓" or "No API key configured" |
| **✦ refined caption strip** | Below feed | Shows Gemini output; hidden when Gemini is off |
| **✦ Gemini is refining...** | Below feed | Shown while API call is in progress |

---

## Getting a Free API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key
5. In the app, click **⚙ Key**, paste the key, and click **Save Key**

> **Note:** Newly created API keys can take 2–3 minutes to become active. If you see a "quota exceeded" error immediately after creating a key, wait a moment and try again.
