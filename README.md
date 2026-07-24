<div align="center">

# 🤟 Sign Language Recognition — Project 3.0

**Real-time hand gesture recognition powered by MediaPipe, OpenCV, and scikit-learn.**

[![Python](https://img.shields.io/badge/Python-3.9%20%E2%80%93%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google&logoColor=white)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f89939?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Capture → Train → Predict in real time using MediaPipe and scikit-learn.

</div>

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [How It Works](#-how-it-works)
3. [File Structure](#-file-structure)
4. [Library Choices & Rationale](#-library-choices--rationale)
5. [Setup & Installation](#️-setup--installation)
6. [Step-by-Step Usage](#-step-by-step-usage)
   - [Step 1 — Collect Data](#step-1--collect-data)
   - [Step 2 — Train the Model](#step-2--train-the-model)
   - [Step 3 — Live Prediction](#step-3--live-prediction)
7. [Technical Deep-Dive](#-technical-deep-dive)
   - [Landmark Extraction & Normalisation](#landmark-extraction--normalisation)
   - [ML Pipeline Architecture](#ml-pipeline-architecture)
   - [Prediction Smoothing & Sentence Builder](#prediction-smoothing--sentence-builder)
8. [UI/UX Design](#-uiux-design)
9. [Performance Tips](#-performance-tips-for-99-accuracy)
10. [Troubleshooting](#️-troubleshooting)
11. [Project Roadmap](#️-project-roadmap)
12. [Contributing](#-contributing)

---

## 🔍 Project Overview

This project lets you **teach a computer to understand sign language in real time** using only a standard webcam — no specialised hardware required. It follows a classic end-to-end machine learning workflow:

```
📷 Webcam  ──▶  🖐 MediaPipe  ──▶  📊 data.csv  ──▶  🧠 sklearn  ──▶  🔮 Live Prediction
```

The system extracts **21 3-D hand landmarks** per frame using Google's MediaPipe, normalises them to be position- and scale-invariant, and feeds the resulting 63-float feature vector into a trained MLP neural network for real-time classification.

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────┐
│  1. collect_data.py                                     │
│     Webcam → MediaPipe → Normalise → Append to CSV      │
├─────────────────────────────────────────────────────────┤
│  2. train_model.py                                      │
│     CSV → Preprocess → CV → Train MLP → Save Pipeline   │
├─────────────────────────────────────────────────────────┤
│  3. predict.py                                          │
│     Webcam → MediaPipe → Normalise → Pipeline → HUD     │
└─────────────────────────────────────────────────────────┘
```

Each script is self-contained and designed to be run in sequence.

---

## 📁 File Structure

```
Sign Language Project 3.0/
│
├── collect_data.py        ← Webcam → landmark extraction → data.csv
├── train_model.py         ← data.csv → ML training → model.joblib
├── predict.py             ← model.joblib → live webcam prediction UI
│
├── data.csv               ← Auto-generated dataset (grows each session)
├── model.joblib           ← Serialised sklearn Pipeline (scaler + MLP)
├── label_classes.joblib   ← LabelEncoder class mapping
│
├── training_report/
│   ├── confusion_matrix.png   ← Heatmap of predictions vs. true labels
│   └── loss_curve.png         ← MLP training/validation loss over epochs
│
├── requirements.txt       ← Python dependencies
└── README.md              ← This file
```

---

## 📦 Library Choices & Rationale

| Library | Role | Why? |
|---|---|---|
| **MediaPipe** | Hand landmark detection | Google's production-grade model; detects 21 3-D landmarks at 30+ FPS with zero manual training. Far more robust than raw contour detection. |
| **OpenCV** | Camera I/O & rendering | Industry standard for real-time video; rich drawing API for the HUD overlay; cross-platform. |
| **Pandas** | Dataset management | `.csv` read/write with named columns; human-readable and version-control friendly (unlike binary formats). |
| **scikit-learn** | ML classifiers & preprocessing | Battle-tested `MLPClassifier`, `StandardScaler`, `Pipeline`, and metrics. Fast to prototype and tune. |
| **Joblib** | Model serialisation | Official sklearn-recommended method; memory-mapped arrays for fast load; safer than Pickle for large NumPy arrays. |
| **NumPy** | Numerical ops | Zero-copy array operations for fast per-frame landmark normalisation. |
| **Matplotlib / Seaborn** | Training visualisation | Confusion matrices and loss curves for diagnosing model quality. |

> **Why CSV over Pickle for data?**  
> CSV files are human-readable, diffable in Git, and can be opened in Excel or Google Sheets. Pickle is binary, version-sensitive, and a security risk if shared. If the landmark schema ever changes, you can inspect and edit rows directly.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.9 – 3.11** *(MediaPipe has limited Python 3.12 support)*
- A working webcam
- Windows / macOS / Linux

### 1 — Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

### 2 — Create a Virtual Environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows tip:** If `mediapipe` fails to install, try:
> ```bash
> pip install mediapipe --extra-index-url https://storage.googleapis.com/mediapipe-releases
> ```

---

## 🚀 Step-by-Step Usage

### Step 1 — Collect Data

```bash
python collect_data.py
```

A window called **"Sign Language — Data Collector"** opens.

| Action | What to do |
|--------|-----------|
| **Start recording a class** | Press any letter key `A`–`Z` |
| **Stop / switch class** | Press another letter key |
| **Quit & save** | Press `Q` or `ESC` |

**What happens automatically:**
1. Your webcam opens and the hand skeleton is overlaid in real time.
2. When you press a key, the script captures **300 frames** for that class.
3. A green progress bar at the bottom tracks collection progress.
4. A blinking **REC** dot indicates active recording.
5. Each completed session is appended to `data.csv` — previous data is never overwritten.

**The CSV schema:**
```
x0, x1, ..., x20, y0, y1, ..., y20, z0, z1, ..., z20, label
-0.12, 0.05, ...                                          , A
```

**Best practices for high accuracy:**
- Collect **≥ 300 frames per class** (the default).
- Run **multiple sessions** at different times to capture natural variation.
- Vary your hand **position, rotation, and distance** within the frame.
- Ensure **good, even lighting** — shadows on the hand cause noisy landmarks.

---

### Step 2 — Train the Model

```bash
python train_model.py
```

The script walks through 8 numbered steps and prints live progress:

| Step | What happens |
|------|-------------|
| 1 | Loads and validates `data.csv` |
| 2 | Encodes labels with `LabelEncoder`, converts features to `float32` |
| 3 | Stratified 80/20 train/test split |
| 4 | 5-fold cross-validation on the training set |
| 5 | Trains MLP (512 → 256 → 128) with early stopping |
| 6 | Prints classification report; saves confusion matrix & loss curve |
| 7 | Benchmarks a Random Forest for comparison |
| 8 | Saves `model.joblib` and `label_classes.joblib` |

**Expected terminal output (example):**

```
  Test Accuracy : 99.8700%

  Classification Report:
               precision    recall  f1-score   support
            A       1.00      1.00      1.00        60
            B       0.99      1.00      1.00        58
            ...
```

**Outputs saved:**
```
model.joblib                          ← sklearn Pipeline (scaler + MLP)
label_classes.joblib                  ← LabelEncoder class array
training_report/confusion_matrix.png  ← per-class accuracy heatmap
training_report/loss_curve.png        ← epoch-by-epoch loss
```

---

### Step 3 — Live Prediction

```bash
python predict.py
```

A window called **"Sign Language — Live Prediction"** opens.

**HUD elements:**

| Element | Location | Description |
|---------|----------|-------------|
| FPS counter | Top-left | Real-time frames per second |
| Predicted letter | Right panel | Large display of the current sign |
| Confidence bar | Right panel | Visual probability fill bar |
| Top-5 predictions | Right panel | Ranked probability bars for the top 5 classes |
| Bounding box | Around hand | Coloured box with L-corner accents |
| Sentence builder | Bottom strip | Auto-appends stable signs into a sentence |
| Stability arc | Right panel | Arc that fills as the sign stabilises |

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `C` | Clear the sentence buffer |
| `SPACE` | Insert a space into the sentence |
| `Q` / `ESC` | Quit |

---

## 🔬 Technical Deep-Dive

### Landmark Extraction & Normalisation

MediaPipe returns 21 landmarks per hand (`x, y, z` each), normalised to the image frame in `[0, 1]`. Raw coordinates are **position-dependent** — the same sign at different screen locations looks different to a naive model.

Two normalisation steps are applied in both `collect_data.py` and `predict.py` (ensuring training/inference consistency):

**1. Translation normalisation** — subtract the wrist (landmark 0):
```python
xs -= xs[0]   # wrist is now at origin (0, 0, 0)
ys -= ys[0]
zs -= zs[0]
```

**2. Scale normalisation** — divide by the bounding-box diagonal:
```python
scale = sqrt((xs.max - xs.min)² + (ys.max - ys.min)²) + 1e-6
xs /= scale
ys /= scale
zs /= scale
```

The resulting 63-float vector is **position-invariant** and **scale-invariant** — the model learns only the *shape* of the gesture (relative finger angles and proportions).

---

### ML Pipeline Architecture

```
Raw features  (63 floats)
      │
      ▼
 StandardScaler          — zero-mean, unit-variance per feature
      │
      ▼
 MLPClassifier
   Hidden layer 1: 512 neurons, ReLU activation
   Hidden layer 2: 256 neurons, ReLU activation
   Hidden layer 3: 128 neurons, ReLU activation
   Output:  softmax  →  class probabilities  [P(A), P(B), …]
      │
      ▼
 predict_proba()  →  top-1 label + confidence score
```

The entire pipeline is wrapped in `sklearn.pipeline.Pipeline`:
- The scaler is **fitted only on training data** — zero data leakage.
- A single `pipeline.predict(X)` handles both scaling and inference.
- `model.joblib` contains both the scaler and the classifier.

---

### Prediction Smoothing & Sentence Builder

Raw per-frame predictions are noisy due to hand tremor and landmark jitter. Two layers of smoothing are applied:

**Layer 1 — Majority-vote buffer** (reduces frame-to-frame flicker):
```python
pred_buffer = deque(maxlen=12)          # last 12 frames
smooth_label = Counter(pred_buffer).most_common(1)[0][0]
```

**Layer 2 — Stability counter** (prevents accidental double-entries):
```python
if smooth_label == stable_label:
    stable_count += 1
if stable_count == 20:                   # 20 consecutive stable frames
    sentence.append(smooth_label)
    stable_count = 0
```

This ensures only **deliberate, held signs** are appended to the sentence.

---

## 🎨 UI/UX Design

The entire HUD is built with OpenCV drawing primitives — no external GUI library required.

| Element | Technique |
|---------|-----------|
| **Rounded rectangle cards** | `cv2.addWeighted` alpha blending over a copied frame |
| **Drop shadows on text** | Text drawn twice — dark offset pass, then coloured foreground |
| **Confidence bars** | `cv2.rectangle` fill proportional to probability |
| **L-corner bounding box** | Four short `cv2.line` segments at each corner |
| **Stability arc** | `cv2.ellipse` with angle proportional to stable frame count |
| **Blinking REC dot** | `int(time.time() * 2) % 2` toggle on a `cv2.circle` |
| **Colour palette** | Dark blue-grey backgrounds with teal/amber accents (BGR) |

---

## 🏆 Performance Tips for >99% Accuracy

1. **Collect more data** — 300+ samples per class across multiple sessions with varied backgrounds and lighting.
2. **Balance your classes** — check `value_counts()` output during training and top up under-represented signs.
3. **Good lighting** — shadows on your hand cause noisy landmarks and degrade accuracy significantly.
4. **Consistent framing** — keep your hand roughly centred and filling ~60–80% of the frame.
5. **Increase MLP capacity** — if accuracy plateaus, try `hidden_layer_sizes=(1024, 512, 256, 128)` in `train_model.py`.
6. **Data augmentation** — mirror coordinate features (`-xs`) to double your dataset for laterally symmetric signs.
7. **Raise confidence threshold** — increase `CONFIDENCE_THRESHOLD` in `predict.py` (e.g., to `0.70`) to suppress uncertain predictions.

---

## 🛠️ Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: mediapipe` | Package not installed | `pip install mediapipe` |
| Webcam shows a black screen | Wrong camera index | Change `CAMERA_INDEX = 1` (or `2`) in the script |
| `FileNotFoundError: data.csv` | No data collected yet | Run `collect_data.py` first |
| `FileNotFoundError: model.joblib` | Model not trained yet | Run `train_model.py` first |
| Hand not detected | Poor lighting or hand too far | Improve lighting; move hand closer to camera |
| Low accuracy (<90%) | Too few samples or only one session | Collect 300+ frames per class across 3+ sessions |
| Predictions flickering | Confidence threshold too low | Increase `CONFIDENCE_THRESHOLD` to `0.70` in `predict.py` |
| `ValueError: class with only N sample(s)` | One class has very few rows | Collect more data for that class |
| Slow FPS (<15) | Heavy CPU load | Close background apps; reduce frame resolution |
| `mediapipe` install error on Python 3.12 | Version incompatibility | Use Python 3.10 or 3.11 instead |

---

## 🗺️ Project Roadmap

- [ ] Pre-trained model for ASL full alphabet (A–Z) + digits (0–9)
- [ ] Two-hand gesture support
- [ ] Text-to-speech output of the predicted sentence
- [ ] Data augmentation (flipping, rotation jitter) built into `train_model.py`
- [ ] Web interface via Flask/FastAPI + WebSocket camera streaming
- [ ] Export to ONNX for edge/mobile deployment
- [ ] Dockerfile for one-command setup

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. **Fork** this repository.
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "Add my feature"`
4. **Push** to your branch: `git push origin feature/my-feature`
5. Open a **Pull Request** and describe what you changed and why.

Please ensure your code:
- Follows the existing style (no type annotations required, but keep it readable).
- Does not break `collect_data.py → train_model.py → predict.py` workflow.
- Includes a brief description in the PR of what was changed and how to test it.

---

<div align="center">

*Built with ❤️ using [MediaPipe](https://mediapipe.dev) · [OpenCV](https://opencv.org) · [scikit-learn](https://scikit-learn.org) · [Joblib](https://joblib.readthedocs.io)*

</div>
