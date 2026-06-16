# 01 — Project Overview

## What is Gestura?

Gestura is an open-source assistive application that translates American Sign Language (ASL) gestures into real-time captions using a webcam and machine learning. The long-term goal is to help Deaf and hard-of-hearing users communicate in video calls (Zoom, Teams, Google Meet) without needing a human interpreter.

---

## Why Gestura Was Built

Video conferencing has become central to professional and social life. For Deaf users, captioning tools like auto-generated subtitles in Zoom exist for spoken audio but not for signed video. Gesture recognition that runs locally, in real time, with no setup beyond installing an app is the gap Gestura aims to fill.

---

## The Journey: v1 → v2

### Gestura v1 — Proof of Concept

Gestura v1 proved that the core idea worked: take a webcam feed, extract skeletal landmarks with MediaPipe, and classify ASL signs using a custom neural network.

| Property | v1 |
|---|---|
| Framework | TensorFlow / Keras |
| Architecture | Conv1D → BiLSTM → BiLSTM → Soft Attention → Softmax |
| Dataset | Self-collected, 17 ASL word classes |
| Samples | 50 sequences × 17 classes = 850 total |
| Sequence shape | `(60, 258)` |
| Accuracy | High on controlled vocab (single signer, consistent lighting) |

**Key learnings from v1:**
- Landmark-based input (not raw RGB) made the model fast, robust to background, and easy to debug
- BiLSTM + attention worked well for a small controlled vocabulary
- The system could not scale: 17 words is not enough for real communication
- There was no sentence output — only the last-predicted word was displayed
- No UI, no virtual camera — it was a raw OpenCV window

### Gestura v2 — Production App

v2 reframes the question: not "can we classify a sign?" but "can we build something a real user would use?". This meant solving four separate engineering problems simultaneously:

1. **Vocabulary** — scale from 17 self-collected words to 300 standardized WLASL classes
2. **Architecture** — replace LSTM with a Transformer that scales better to 300 classes
3. **App** — build a PyQt6 desktop app with multi-threading, UI controls, and virtual camera output
4. **Polish** — temporal smoothing, fingerspelling fallback, and LLM caption correction

---

## v1 vs v2 Comparison

| Area | v1 | v2 |
|---|---|---|
| Goal | Research prototype | Assistive production app |
| Framework | TensorFlow / Keras | PyTorch |
| Model | Conv1D + BiLSTM + Attention | Conv1D + Transformer (GestureTransformer) |
| Vocabulary | 17 self-collected words | 300 WLASL word classes |
| Dataset size | 850 sequences | 3,667 videos |
| UI | OpenCV window | PyQt6 desktop app |
| Output | Raw word label | Live subtitle overlay + virtual camera feed |
| Fallback | None | A–Z Fingerspelling (Layer 2) |
| Grammar correction | None | Gemini AI (Layer 3) |
| Threading | Single-threaded | 4 worker threads (capture, inference, UI, vcam) |
| Camera robustness | None | Backend fallback (Default → DirectShow) |

---

## Vocabulary Covered (300 WLASL Words)

Common words include: `about`, `again`, `all`, `apple`, `baby`, `bad`, `ball`, `because`, `birthday`, `black`, `blue`, `book`, `boy`, `bring`, `brother`, `buy`, `can`, `cat`, `chair`, `change`, `children`, `close`, `coffee`, `cold`, `color`, `computer`, `cook`, `dance`, `day`, `deaf`, `different`, `doctor`, `dog`, `door`, `drink`, `drive`, `easy`, `eat`, `enjoy`, `family`, `far`, `father`, `feel`, `fine`, `finish`, `fish`, `flower`, `forget`, `friend`, `game`, `girl`, `give`, `go`, `good`, `happy`, `hard`, `have`, `hear`, `heart`, `help`, `here`, `home`, `hot`, `house`, `how`, `hurry`, `husband`, `jump`, `kill`, `kiss`, `know`, `language`, `late`, `laugh`, `learn`, `leave`, `like`, `live`, `lose`, `make`, `man`, `many`, `mean`, `meet`, `milk`, `money`, `more`, `mother`, `movie`, `music`, `name`, `need`, `new`, `no`, `now`, `old`, `order`, `paint`, `party`, `person`, `pizza`, `plan`, `play`, `please`, `practice`, `problem`, `read`, `red`, `remember`, `restaurant`, `right`, `run`, `school`, `share`, `show`, `sick`, `sign`, `small`, `some`, `soon`, `stay`, `student`, `study`, `take`, `teach`, `teacher`, `tell`, `test`, `time`, `tired`, `train`, `travel`, `visit`, `wait`, `walk`, `want`, `water`, `week`, `what`, `where`, `who`, `why`, `wife`, `window`, `with`, `woman`, `work`, `write`, `wrong`, `year`, `yes`, `you`, `your`, *(and ~150 more)*.
