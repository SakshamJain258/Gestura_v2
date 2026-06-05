# Gestura Project Deep Dive

This document prepares both versions of Gestura for resumes, interviews, project demos, and technical discussion.

## High-Level Evolution

Gestura v1 was the proof-of-concept: a real-time webcam sign recognizer built around MediaPipe skeletal landmarks and a temporal neural network. It proved that landmark sequences could be captured, trained, and recognized live with high accuracy on a controlled vocabulary.

Gestura v2 is the productization step: a desktop assistive app for video calls. It expands the vocabulary from a small/self-collected set to WLASL-300, moves the research model to PyTorch, introduces a Transformer-based recognizer, and adds a production-style real-time pipeline with capture, inference, smoothing, caption rendering, and virtual camera output.

## Version 1: Real-Time Sign Language Translator

### One-Line Pitch

Gestura v1 is a real-time ASL recognition prototype that converts webcam video into MediaPipe landmark sequences and classifies signs using a custom temporal model with attention.

### Problem It Solved

The goal was to recognize signs live from a webcam without using pre-built sign classifiers. Instead of classifying raw images directly, the system extracts body and hand landmarks, converts each gesture into a time-series sequence, and predicts the sign from motion patterns.

### Pipeline

1. Webcam frames are captured with OpenCV.
2. Frames are mirrored for natural interaction.
3. MediaPipe Holistic detects pose, left-hand, and right-hand landmarks.
4. Each frame is converted into a 258-dimensional feature vector:
   - Pose: 33 landmarks x 4 values = 132 features
   - Left hand: 21 landmarks x 3 values = 63 features
   - Right hand: 21 landmarks x 3 values = 63 features
5. A gesture sample is stored as 60 frames x 258 features.
6. The model predicts one sign class from the 60-frame sequence.

### Dataset

From the code, v1 uses a self-collected landmark dataset under `data2`, with 50 recorded sequences per class and 60 frames per sequence. The visible class list contains 17 classes:

`goodbye`, `hello`, `yes`, `no`, `thank you`, `please`, `sorry`, `stop`, `help`, `what`, `how`, `where`, `when`, `eat`, `drink`, `sleep`, `IDLE_STATE`

If your resume says 26 ASL signs, be ready to explain whether that refers to an earlier alphabet model/notebook or a different final training run. The Python file currently visible is a word-level recognizer, not a 26-letter alphabet recognizer.

### Model Architecture

The inspected v1 code uses TensorFlow/Keras, not PyTorch. Its current architecture is:

1. `Conv1D(256, kernel_size=3)` to learn local temporal motion patterns.
2. Batch normalization and dropout for stability and regularization.
3. Two bidirectional LSTM layers:
   - BiLSTM with 128 units
   - BiLSTM with 64 units
4. Soft temporal attention:
   - A dense layer scores each timestep.
   - Softmax normalizes frame importance across the sequence.
   - Attention weights multiply the LSTM outputs.
   - Weighted timestep features are summed into one context vector.
5. Dense classification head.
6. Softmax output over sign classes.

### Why This Design Makes Sense

Conv1D captures short motion fragments such as hand opening, wrist movement, and quick transitions. BiLSTM captures how the sign evolves over time in both directions. Attention helps the model focus on the most discriminative frames, because not every frame in a 60-frame recording is equally useful. Some frames are setup, some are hold, and some contain the actual distinguishing movement.

### Strong Interview Explanation

"I represented each sign as a temporal sequence of skeletal landmarks rather than raw RGB frames. That made the model lighter and more robust to background, lighting, and clothing. I used MediaPipe only for landmark extraction, not for classification. The classifier itself was trained from my collected sequences. The model combines Conv1D for local motion features, BiLSTM for sequence modeling, and soft attention so the network can weight important frames more heavily than noisy or transitional frames."

### Technical Challenges

- Collecting consistent gesture samples while maintaining natural signing motion.
- Handling missing landmarks when one hand leaves the frame.
- Avoiding flickering predictions during live inference.
- Choosing sequence length: long enough to capture full gestures, short enough for real-time use.
- Distinguishing visually similar signs where the difference is in timing or hand trajectory.

### Resume Wording Suggestion

Use this if v1 is based on the inspected code:

`Designed and trained a custom Conv1D-BiLSTM attention model on MediaPipe pose/hand landmark sequences for real-time ASL word recognition from webcam input, using no pre-built sign classifier.`

Use this only if you truly have a PyTorch alphabet model elsewhere:

`Designed and trained a custom PyTorch CNN-LSTM model from scratch to classify 26 ASL alphabet signs from live webcam skeletal landmark sequences.`

## Version 2: Assistive ASL-to-Caption Desktop App

### One-Line Pitch

Gestura v2 is a production-oriented desktop app that translates live ASL gestures into captions and streams the annotated output as a virtual camera for Zoom, Teams, and Google Meet.

### Product Goal

v2 moves from "can I classify a sign?" to "can this help someone communicate in a real video call?" That changes the engineering problem. The system must be responsive, stable, multi-threaded, usable from a GUI, and compatible with external meeting software.

### System Architecture

The v2 system is organized around independent real-time workers:

1. `CaptureThread`
   - Reads frames from the webcam.
   - Emits frames quickly without running heavy inference.
   - Keeps capture responsive even when model inference is slower.

2. `InferenceThread`
   - Keeps only the freshest frame using a small queue.
   - Runs MediaPipe Holistic.
   - Extracts 258-dimensional landmark vectors.
   - Maintains a rolling 60-frame sequence.
   - Runs the PyTorch GestureTransformer.
   - Applies confidence thresholding and temporal smoothing.

3. `TemporalSmoother`
   - Reduces flicker by requiring repeated stable predictions.
   - Avoids adding every noisy frame-level prediction to the caption.

4. `VirtualCamThread`
   - Sends annotated frames to `pyvirtualcam`.
   - Runs separately because `pyvirtualcam.send()` can block.
   - Prevents meeting-app output from slowing model inference.

5. PyQt6 UI
   - Displays camera preview, captions, status, FPS, and threshold controls.
   - Provides a usable desktop shell around the recognition system.

### v2 Model: GestureTransformer

The v2 model is a PyTorch Conv1D + Transformer Encoder architecture for WLASL-300 word recognition.

Input:

`(batch, 60 frames, 258 landmark features)`

Main components:

1. Structured landmark embedding:
   - Pose, left hand, and right hand are projected separately.
   - This preserves anatomical structure before feature fusion.

2. Multi-scale Conv1D temporal block:
   - Parallel temporal kernels of size 3, 5, and 7.
   - Captures short, medium, and slightly longer motion patterns.
   - Fuses them with a 1D convolution.

3. Positional encoding:
   - Adds frame-order information.
   - Important because Transformers do not naturally know sequence order.

4. Learnable CLS token:
   - Prepended to the sequence.
   - Used as the final summary representation for classification.

5. Transformer Encoder:
   - Multi-head self-attention captures global temporal relationships across all 60 frames.
   - Better suited than LSTM for scaling to 300 classes because attention can compare all timesteps directly.

6. Classification head:
   - LayerNorm, linear projection, GELU, dropout, and final 300-class output.

### WLASL-300 Dataset Work

The v2 training pipeline converts raw WLASL videos into landmark arrays:

1. Load WLASL videos.
2. Run MediaPipe Holistic on each frame.
3. Extract pose and hand landmarks.
4. Pad or truncate each sample to 60 frames.
5. Save each sample as `.npy` with shape `(60, 258)`.
6. Build `manifest.json` and `label_map.json`.

Your resume numbers say 3,667 videos across 300 ASL word classes. That is consistent with the WLASL-300 direction in the docs.

### Training Strategy

The training script includes several good engineering choices:

- PyTorch training loop for research flexibility.
- CUDA/mixed precision support for faster training.
- AdamW optimizer with weight decay.
- Cosine annealing learning-rate schedule.
- Class-weighted cross entropy for class imbalance.
- Label smoothing to reduce overconfidence.
- Mixup to improve generalization on small class counts.
- Data augmentation:
  - Temporal jitter
  - Gaussian landmark noise
  - Frame dropout
  - Time warping
  - Left/right mirror flip
  - Hand scaling
  - Spatial jitter
- Top-1 and top-5 validation tracking.

### Accuracy Interpretation

The reported `39.6% top-1` and `60.2% top-5` validation accuracy should be framed carefully. For 300 classes with limited samples per word, this is much harder than a small self-collected vocabulary. Top-5 matters because similar signs can be visually close, and later language/context layers can use candidate lists to choose a better sentence.

Strong framing:

`On WLASL-300, the model achieved 39.6% top-1 and 60.2% top-5 validation accuracy across 300 word classes, which is a substantially harder open-vocabulary setting than v1's controlled webcam dataset.`

### Three-Layer Recognition Pipeline

The planned v2 recognition stack is:

1. Gesture model:
   - Handles the main 300-word vocabulary.
   - Produces word probabilities and confidence.

2. Fingerspelling fallback:
   - Triggered when the gesture model confidence is low.
   - Recognizes A-Z letters for names, places, and out-of-vocabulary words.

3. LLM correction layer:
   - Converts raw ASL-style word streams into natural English captions.
   - Example: `hello you name what` -> `Hello, what is your name?`

This layered design is strong because each layer has a clear job. The gesture model recognizes signs; the fingerspelling model handles vocabulary gaps; the LLM fixes grammar and fluency.

### Strong Interview Explanation

"The biggest lesson from v1 was that sign recognition is not only a model problem. For a real assistive app, latency and reliability matter just as much as accuracy. In v2 I separated capture, inference, smoothing, UI, and virtual camera output into different threads. That way, blocking calls from meeting software do not stall inference. On the ML side, I moved from a BiLSTM attention model to a PyTorch GestureTransformer trained on WLASL-300. It uses structured landmark embeddings, multi-scale Conv1D for local motion, and Transformer attention for global temporal reasoning."

## Version 1 vs Version 2

| Area | Version 1 | Version 2 |
|---|---|---|
| Goal | Real-time recognition prototype | Assistive caption desktop app |
| Input | Webcam landmarks | Webcam/video-call landmarks |
| Dataset | Self-collected landmark sequences | WLASL-300 landmark sequences |
| Vocabulary | Small controlled set / possibly alphabet depending on run | 300 ASL word classes |
| Framework | TensorFlow/Keras in inspected code | PyTorch |
| Model | Conv1D + BiLSTM + attention | Structured embedding + Conv1D + Transformer |
| Output | Predicted sign/word | Live caption + virtual camera feed |
| Engineering focus | Data collection and live prediction | Multi-threading, smoothing, virtual cam, app UX |

## Best Demo Story

Start with v1:

"I first proved the recognition idea on a controlled webcam dataset. I collected landmark sequences, trained a custom temporal model, and got live predictions working."

Then transition:

"But a prototype is not enough for real communication. Real users need captions in video calls, support for more words, stable output, and fallback behavior for unknown words. That led to v2."

Then explain v2:

"v2 expands the model to WLASL-300 with a Transformer architecture and wraps it in a PyQt6 desktop app. The app has separate threads for webcam capture, inference, smoothing, and virtual-camera output so it stays responsive while streaming captions into Zoom/Teams/Meet."

## Questions You Should Be Ready For

### Why landmarks instead of raw video?

Landmarks reduce input dimensionality massively, remove background noise, and make the system easier to run in real time. A raw video model would need more data, more compute, and would overfit more easily to lighting, camera, and signer appearance.

### Why LSTM in v1?

Because signs are temporal. A single frame does not capture motion. LSTM models how landmark positions evolve over time, while BiLSTM lets the model use both earlier and later context within the fixed recorded sequence.

### Why attention?

A 60-frame gesture contains setup frames, transition frames, and key discriminative frames. Attention lets the model assign higher weight to the frames that matter most for classification.

### Why Transformer in v2?

Transformers scale better for a larger vocabulary because self-attention compares all timesteps directly. They can learn long-range dependencies and subtle relationships between joints across the full sequence without recurrence.

### Why top-5 accuracy?

In a 300-class sign model, top-1 is strict. Top-5 tells whether the correct sign is among the model's strongest candidates, which is valuable when a later language correction layer can use context to resolve ambiguity.

### Why a fingerspelling fallback?

No fixed word vocabulary can cover names, places, acronyms, and technical terms. Fingerspelling is how signers handle many out-of-vocabulary terms, so the app needs it to avoid breaking outside the 300-word set.

### Why an LLM correction layer?

ASL word order is not identical to English word order. The recognizer may output a raw sign stream like `you name what`; the LLM layer turns that into natural English captions like `What is your name?`

## Resume Cautions

There are two important consistency checks before you finalize resume bullets:

1. v1 framework mismatch:
   - Your bullet says PyTorch.
   - The inspected v1 code uses TensorFlow/Keras.
   - Fix this unless there is another PyTorch implementation you plan to present.

2. v1 class-count mismatch:
   - Your bullet says 26 ASL signs.
   - The inspected v1 `ASL.py` list contains word classes and `IDLE_STATE`.
   - If 26 refers to alphabet recognition, keep that as a separate version/run and be ready to show the matching notebook/code.

## Clean Resume Version

### Version 1

`Gestura v1 - Real-Time ASL Recognition Prototype`

`Python, TensorFlow/Keras, Conv1D, BiLSTM, Soft Attention, OpenCV, MediaPipe, NumPy`

- Built a real-time ASL recognition prototype that converts webcam video into 60-frame MediaPipe pose/hand landmark sequences and classifies gestures using a custom temporal neural network.
- Designed a Conv1D + BiLSTM + soft-attention model to capture local motion, sequence dynamics, and discriminative frames while filtering noisy transitions.
- Created a custom data-collection and sequencing pipeline using OpenCV and MediaPipe, storing each gesture as `(60, 258)` landmark tensors for supervised training and live inference.

### Version 2

`Gestura v2 - Assistive ASL-to-Caption Desktop App for Video Calls`

`Python, PyTorch, Conv1D + Transformer, PyQt6, MediaPipe, OpenCV, pyvirtualcam, TensorFlow`

- Building a production-oriented desktop app that translates live ASL into captions and streams annotated video through a virtual camera for Zoom, Teams, and Google Meet.
- Designed and trained a GestureTransformer with structured landmark embeddings, multi-scale Conv1D temporal features, Transformer encoder attention, and a 300-class WLASL output head.
- Trained on WLASL-300 landmark sequences from 3,667 videos across 300 ASL word classes, achieving 39.6% top-1 and 60.2% top-5 validation accuracy.
- Engineered a real-time multi-threaded pipeline separating webcam capture, MediaPipe inference, temporal smoothing, UI updates, and virtual-camera output to reduce latency and prediction flicker.
- Designed a layered recognition strategy: 300-word gesture model, confidence-triggered A-Z fingerspelling fallback, and LLM correction for converting raw ASL word streams into natural English captions.

