# Gestura

Gestura is an evolving project aimed at providing real-time American Sign Language (ASL) recognition and translation. It is designed to act as an assistive application that translates live ASL gestures into captions and streams the annotated output as a virtual camera for video conferencing tools like Zoom, Teams, and Google Meet.

Currently, Gestura is in active development, transitioning from our initial proof-of-concept (v1) to a robust, production-oriented desktop application (v2).

## The Journey: Why Gestura v2?

### Gestura v1: The Proof of Concept (Legacy)
Our initial version, **Gestura v1**, was a real-time ASL recognition prototype. It proved that we could convert webcam video into skeletal landmarks using MediaPipe and classify signs live using a custom temporal neural network. 
- **Tech Stack**: TensorFlow/Keras, Conv1D, BiLSTM, Soft Attention.
- **Scope**: Controlled vocabulary (17 self-collected words) for live prediction.
- **Takeaway**: It worked well for a controlled set, but real-world communication requires more vocabulary, better stability, and integration with meeting software.

### Gestura v2: The Production App (Active Development)
**Gestura v2** represents the productization step. We moved from asking "can we classify a sign?" to "can we build a reliable assistive tool for real video calls?" 

**Why v2?**
- A prototype is not enough for real communication. Users need captions in video calls, support for more words, stable output without flickering, and fallback behavior for unknown words.
- To handle the increased complexity and vocabulary, we migrated to **PyTorch** and a **Transformer-based** architecture, expanding our target vocabulary to the WLASL-300 dataset (300 ASL word classes).
- We engineered a multi-threaded desktop app to separate webcam capture, inference, temporal smoothing, and virtual-camera output. This ensures the app remains responsive even when streaming to external software.

## What We Have Achieved So Far (v2 Progress)

We have successfully laid the groundwork for the v2 pipeline and are currently iterating on the models and UI:

- **WLASL-300 Dataset Pipeline**: Built a pipeline to convert 3,667 raw WLASL videos into `(60, 258)` landmark tensors using MediaPipe Holistic.
- **GestureTransformer Model**: Designed and trained a PyTorch Conv1D + Transformer Encoder model. It uses structured landmark embeddings, multi-scale Conv1D for local motion, and Transformer attention for global temporal reasoning.
- **Initial Training Results**: On the challenging 300-class WLASL dataset, the model has achieved **39.6% top-1** and **60.2% top-5** validation accuracy (a significant milestone for an open-vocabulary setting).
- **Multi-threaded PyQt6 App**: Developed the core desktop application with separate threads for `CaptureThread`, `InferenceThread`, `TemporalSmoother`, and `VirtualCamThread`, preventing blocking calls from stalling inference.
- **Layered Recognition Strategy (Planned/In Progress)**:
  1. Main gesture model for 300 words.
  2. A confidence-triggered A-Z fingerspelling fallback (for names/out-of-vocabulary).
  3. An LLM correction layer to convert raw ASL streams (`hello you name what`) to natural English (`Hello, what is your name?`).

> **Note:** Gestura v2 is **still a work in progress**. We are actively refining the model accuracy, integrating the fingerspelling fallback, and polishing the user interface.

---

## Setup & Run Instructions (Gestura v2)

### Prerequisites
- Python 3.9+
- A working webcam

### Setup Environment

**For CPU/default PyTorch:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**For CUDA PyTorch (Recommended for faster training/inference):**
*Install the CUDA wheel first, then install the remaining requirements:*
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### Run the App

```powershell
cd app
python app.py
```

### Project Structure

- `app/` contains the runnable Phase 1 PyQt6 desktop app.
- `training/` contains training code, data processing scripts, selected results, and trained PyTorch model assets.
- `workflow/` contains workflow/reference assets.
- `Gestura_Project_Deep_Dive.md` contains detailed architectural explanations and interview prep material.

*(Note: Keep `.venv/` local. It is ignored by Git.)*
