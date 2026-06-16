<h1 align="center">
  <br>
  🤟 Gestura v2
  <br>
</h1>

<h4 align="center">Real-time ASL gesture recognition that renders live captions directly onto your video feed — no screen sharing, no typing, just sign.</h4>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-0.10.14-00A58E?style=flat-square">
  <img alt="PyQt6" src="https://img.shields.io/badge/PyQt6-Desktop-41CD52?style=flat-square&logo=qt">
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square">
</p>

<p align="center">
  <a href="#-what-is-gestura">What is it?</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#%EF%B8%8F-model">Model</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

---

## 🧏 What is Gestura?

Gestura is an open-source assistive desktop application for Windows that translates **American Sign Language (ASL) gestures into real-time captions** using a webcam and machine learning — entirely locally, with no cloud processing of your video.

The captions are **burned directly onto your video feed** and piped as a **virtual camera** ("Gestura Cam"). Anyone on your Zoom, Google Meet, or Microsoft Teams call sees your face + captions without needing any extra software on their end.

**Built for:** Specially-abled, non-vocal users who want to communicate in video calls without a human interpreter.

---

## 🎬 Demo

▶️ **[Watch the demo on YouTube](https://www.youtube.com/watch?v=-pgvQyp-oF0)**

📄 **[Research Paper](https://drive.google.com/file/d/1zT-9kl8uMixt35fh_YmMm7v4dmrmR6jh/view?usp=sharing)**

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Vocabulary** | 300 ASL word classes (WLASL-300 dataset) |
| **Real-time recognition** | MediaPipe → GestureTransformer inference at ~25 FPS |
| **Virtual camera output** | Pushes captioned feed as "Gestura Cam" at 1280×720, 30 FPS |
| **Fingerspelling fallback** | A–Z letter recognition for names and out-of-vocabulary words |
| **LLM grammar correction** | Optional Gemini API layer converts raw word streams to natural English |
| **Temporal smoothing** | Majority voting over 6-frame window prevents subtitle flickering |
| **Offline capable** | Core recognition runs fully offline — Gemini is opt-in |
| **Multi-threaded** | 4 parallel threads: capture, inference, UI, virtual camera |

---

## 🧠 How It Works

Gestura runs three layers of AI in sequence:

```
Your Webcam
    │
    ▼ Layer 1 — Gesture Recognition
    │  MediaPipe Holistic extracts 258 skeletal landmarks per frame
    │  (33 pose joints + 21 left-hand + 21 right-hand joints)
    │  GestureTransformer (1.66M params) analyzes 60-frame windows
    │  → Predicts 1 of 300 ASL words
    │
    ▼ Layer 2 — Fingerspelling Fallback
    │  Lightweight MLP reads single right-hand frames
    │  → Predicts A–Z letters for names and rare words
    │
    ▼ Layer 3 — LLM Grammar Correction (optional)
    │  After a signing pause (2.5 sec), raw word stream is sent to Gemini API
    │  → Converts "you name what" → "What is your name?"
    │
    ▼ Output
       Captions burned onto video frames → pyvirtualcam → "Gestura Cam"
       (Select "Gestura Cam" in Zoom / Meet / Teams)
```

### Threading Model

```
Main Thread (Qt)  ←─ signals ──────────────────────────────────────┐
                                                                     │
CaptureThread  ──frame_ready──►  InferenceThread  ──push_frame──►  VirtualCamThread
(cv2.VideoCapture)               (MediaPipe + Transformer)          (pyvirtualcam)
```

---

## 🚀 Getting Started

### Prerequisites

- **Windows 10 or 11**
- **Python 3.11**
- A **webcam**
- **4 GB RAM** minimum
- *(Optional)* NVIDIA GPU for faster inference
- *(Optional)* [OBS Studio](https://obsproject.com/) for the virtual camera feature

---

### 1. Clone the Repository

```powershell
git clone https://github.com/SakshamJain258/Gestura_v2.git
cd Gestura_v2
```

---

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

---

### 3. Install Dependencies

**CPU only (works on any machine):**
```powershell
pip install -r requirements.txt
```

**CUDA (recommended if you have an NVIDIA GPU):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> ⚠️ **Pinned versions matter.** Do not upgrade `mediapipe`, `opencv-contrib-python`, `protobuf`, or `numpy` — the versions in `requirements.txt` are the only validated combination.

---

### 4. Download the Model Weights

The trained model weights are not included in this repository due to file size. Download them from [**GitHub Releases**](https://github.com/SakshamJain258/Gestura_v2/releases) and place them as follows:

```
Gestura_v2/
├── gesture_model_300_inference.pt    ← word recognition model (~6.7 MB)
└── fingerspelling_model.pt           ← A–Z fingerspelling model
```

---

### 5. *(Optional)* Set Up Virtual Camera

The virtual camera pipes Gestura's annotated feed into your meeting app.

1. Install **[OBS Studio](https://obsproject.com/)**
2. Open OBS → **Tools → Virtual Camera → Start Virtual Camera** (do this once to register the driver)
3. In your meeting app, select **"OBS Virtual Camera"** as your camera input

> **Without a virtual camera driver:** Gestura still runs — the captions show in the app window, but won't appear in your meeting feed.

---

### 6. Run the App

```powershell
cd app
python app.py
```

Then:
1. Click **Start** in the app
2. Open your meeting app and select **"OBS Virtual Camera"** as your camera
3. Start signing — your captions appear live for everyone on the call

---

### 7. *(Optional)* Enable Gemini Grammar Correction

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. In the app, click **⚙ Key** → paste your key → **Save Key**
3. Toggle **Enable Gemini** on

> Newly created keys take 2–3 minutes to activate.

---

## 🏛️ Model

### GestureTransformer

A custom Transformer architecture trained on 3,667 videos from the WLASL-300 dataset.

```
Input: (batch, 60 frames, 258 landmarks)

LandmarkEmbedding        — projects pose/hands separately, then fuses
    │
ConvTemporalBlock        — multi-scale Conv1D (k=3, 5, 7) for local motion
    │
CLS token + Positional Encoding
    │
TransformerEncoder × 3   — 6-head attention, pre-LayerNorm
    │
CLS token → Classifier head
    │
Output: (batch, 300) logits
```

| Parameter | Value |
|---|---|
| Total parameters | ~1.66M |
| d_model | 192 |
| Attention heads | 6 |
| Encoder layers | 3 |
| Input sequence | 60 frames × 258 features |
| Output classes | 300 ASL words |

### FingerspellingClassifier

A lightweight MLP for single-frame A–Z letter recognition.

```
Input: (63,) — right-hand 21 landmarks × (x, y, z)
Linear(63→256) → ReLU → Dropout(0.3)
Linear(256→128) → ReLU → Dropout(0.3)
Linear(128→27)
Output: 27 logits (A–Z + space)
```

### Training

| Setting | Value |
|---|---|
| Dataset | WLASL-300 (~3,667 videos, 300 classes) |
| Optimizer | AdamW |
| Learning rate | 5e-4 → 1e-6 (CosineAnnealingLR) |
| Epochs | 500 |
| Best Val Top-1 | 6.78% |
| Best Val Top-5 | 23.57% |
| Train-Val Gap | 1.16% |

> **On accuracy:** 6.78% top-1 on 300 classes with ~8 videos per class is expected — published academic models without pre-training report similar numbers on this benchmark. The model's real strength is temporal smoothing + fingerspelling fallback in the live app.

---

## 📁 Project Structure

```
Gestura_v2/
├── app/                          # Desktop application
│   ├── app.py                    # Entry point
│   ├── core/
│   │   ├── inference_assets.py   # GestureTransformer + inference utilities
│   │   ├── caption_buffer.py     # Rolling word buffer + pause detection
│   │   ├── smoother.py           # Majority-vote temporal smoother
│   │   ├── mode_controller.py    # Word ↔ fingerspelling mode switching
│   │   ├── pause_detector.py     # Signing pause → Gemini trigger
│   │   ├── llm_client.py         # Gemini API wrapper
│   │   ├── fingerspelling_model.py
│   │   └── startup_checks.py     # Pre-launch validation
│   ├── threads/
│   │   ├── capture_thread.py     # Webcam acquisition
│   │   ├── inference_thread.py   # MediaPipe + model inference
│   │   └── virtual_cam_thread.py # pyvirtualcam output
│   └── ui/
│       ├── main_window.py        # PyQt6 main window
│       └── api_key_dialog.py     # Gemini key dialog
│
├── training/                     # Training scripts (not needed to run the app)
│   ├── Model/
│   │   ├── model.py              # GestureTransformer definition
│   │   ├── train.py              # Training loop
│   │   └── extract_landmarks.py  # WLASL video → .npy landmark extraction
│   └── fingerspelling/
│       ├── train_fingerspelling.py
│       └── generate_synthetic_landmarks.py
│
├── docs/                         # Detailed technical documentation
│   ├── 01_project_overview.md
│   ├── 02_dataset_pipeline.md
│   ├── 03_model_architecture.md
│   ├── 04_training_strategy.md
│   ├── 05_app_architecture.md
│   ├── 06_camera_pipeline.md
│   ├── 07_llm_integration.md
│   └── 08_deployment_and_build.md
│
├── landmarks_300/                # Extracted .npy landmark files (gitignored)
├── WLASL_300/                    # Raw video dataset (gitignored, ~18 GB)
├── requirements.txt
└── README.md
```

---

## ⚙️ Key Dependency Versions

These versions are **pinned** — changing them (especially MediaPipe, protobuf, numpy) will likely break the app.

| Package | Version |
|---|---|
| `mediapipe` | `0.10.14` |
| `opencv-contrib-python` | `4.8.1.78` |
| `protobuf` | `4.25.3` |
| `numpy` | `1.26.4` |
| `torch` | latest 2.x |
| `PyQt6` | latest |
| `pyvirtualcam` | latest (optional) |
| `google-generativeai` | latest (optional) |

---

## 🔧 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Model file not found` | `.pt` file missing | Download from GitHub Releases and place in repo root |
| Camera feed blank, no error | OpenCV default backend silent failure | App auto-falls back to DirectShow. If still blank, close Teams/Zoom (they lock the camera) |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Wrong MediaPipe version | `pip install mediapipe==0.10.14` |
| `protobuf version mismatch` | Version conflict | `pip install protobuf==4.25.3` |
| `ImportError: PyQt6` | PyQt6 not installed | `pip install PyQt6` |
| Gemini toggle greyed out | No API key configured | Click **⚙ Key** in the app |
| Camera locked after a crash | Orphaned Python process holding device handle | `Stop-Process -Name python -Force` in PowerShell |
| Subtitles flickering | Confidence threshold too low | Raise the confidence slider in the app |
| No hand tracking | Hands not visible / bad lighting | Enable **Show Landmarks Overlay** to debug — ensure both hands are in frame with even lighting |

---

## 📖 Technical Documentation

Detailed writeups for each subsystem live in [`docs/`](./docs):

| Doc | Contents |
|---|---|
| [01 — Project Overview](./docs/01_project_overview.md) | v1 → v2 journey, feature comparison |
| [02 — Dataset Pipeline](./docs/02_dataset_pipeline.md) | WLASL-300, landmark extraction, feature vectors |
| [03 — Model Architecture](./docs/03_model_architecture.md) | GestureTransformer + FingerspellingClassifier |
| [04 — Training Strategy](./docs/04_training_strategy.md) | Augmentation, loss, optimizer, training runs |
| [05 — App Architecture](./docs/05_app_architecture.md) | Threading model, signal flow, component design |
| [06 — Camera Pipeline](./docs/06_camera_pipeline.md) | Windows camera bugs + fixes |
| [07 — LLM Integration](./docs/07_llm_integration.md) | Gemini caption refinement architecture |
| [08 — Deployment & Build](./docs/08_deployment_and_build.md) | Running from source, PyInstaller plans |

---

## 🗺️ Roadmap

| Status | Feature |
|---|---|
| ✅ Shipped | 300-word ASL gesture recognition (WLASL-300) |
| ✅ Shipped | PyQt6 desktop app with multi-threading |
| ✅ Shipped | Virtual camera output ("Gestura Cam") |
| ✅ Shipped | Temporal smoothing + confidence threshold |
| ✅ Shipped | A–Z fingerspelling fallback layer |
| ✅ Shipped | Gemini LLM grammar correction layer |
| ✅ Shipped | Standalone Windows `.exe` (PyInstaller) |
| 📋 Planned | Expanded vocabulary beyond 300 words |
| 📋 Planned | macOS / Linux support |

---

## 🌐 Website & Landing Page

The web portal and landing page for **Gestura** are hosted in a separate repository. This website serves as the presentation portal and download distributor for the packaged desktop application.

🔗 **Website Repository:** [Gestura Web Portal](https://github.com/SakshamJain258/Gestura-Website) *(Note: Link will be updated as the separate repo is published.)*

---

## 🙌 Acknowledgements

- **[WLASL Dataset](https://github.com/dxli94/WLASL)** — Dongxu Li et al. (2020) — the benchmark ASL video dataset used for training
- **[MediaPipe](https://github.com/google/mediapipe)** — Google's real-time skeletal landmark extraction
- **[pyvirtualcam](https://github.com/letmaik/pyvirtualcam)** — virtual camera output library
- **[Google Gemini API](https://aistudio.google.com/)** — LLM grammar correction layer

---

## 👨‍💻 Author

**Saksham Jain** — 3rd Year CS Student, Bennett University (2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/saksham-jain-6a74b128a/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/SakshamJain258)

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

<p align="center">
  <em>Built with obsession for building things that matter. 🤟</em>
</p>

