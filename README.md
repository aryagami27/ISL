# Indian Sign Language (ISL) Recognition System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed_by-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32+-00C853.svg)](https://developers.google.com/mediapipe)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-99.95%25-brightgreen.svg)]()

An end-to-end computer vision and deep learning system for high-accuracy Indian Sign Language (ISL) gesture recognition across **35 alphanumeric classes** (Digits `1–9` and Letters `A–Z`).

The repository features three distinct architectural approaches, headlined by a state-of-the-art **3D Dual-Hand Bone Geometry & Landmark Cutout Neural Network** achieving **99.95% accuracy** with sub-3ms CPU inference latency.

---

## Request Implementation Status

| Step | Description | Target / Deliverable | Status |
| :--- | :--- | :--- | :--- |
| **Step 1** | Audit Codebase & Project Architecture | Inspected scripts, dependencies (`pyproject.toml`), datasets, and checkpoints | `[x] Completed` |
| **Step 2** | Create Agent Documentation Hub (`agent/README.md`) | Central navigation index linking all internal architecture docs | `[x] Completed` |
| **Step 3** | Document Project Structure (`agent/project_structure.md`) | File tree, component breakdown, module dependencies, and data flow | `[x] Completed` |
| **Step 4** | Document Logic & Mathematics (`agent/logic_implementation.md`) | 3D normalization, bone vectors, dot-product angles, and Landmark Cutout | `[x] Completed` |
| **Step 5** | Document Trial & Decision History (`agent/trial_history.md`) | Chronological evolution from 2D CNNs to 3D Landmark Geometry & benchmarks | `[x] Completed` |
| **Step 6** | Document Operational Facts (`agent/established_facts.md`) | Verified specs, hardware acceleration, caching policies, and port bindings | `[x] Completed` |
| **Step 7** | Standardize Root README (`README.md`) | Structured Setup, Running, Documentation, and Quick Start guides | `[x] Completed` |

---

## Model Comparison

| # | Script | Approach | Backbone / Architecture | Val Accuracy | Latency (CPU) | Size |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: |
| **1** | `model1_ultralight.py` | Image 2D CNN | MobileNetV2 (ImageNet pretrained) | ~88.5% | ~30 ms | 10.2 MB |
| **2** | `model2_highperf.py` | Image 2D Deep CNN | ResNet50 (Transfer Learning) | ~93.8% | ~90 ms | 95.3 MB |
| **⭐ 3** | **`model_best.py`** | **3D Landmark + Bone Geometry** | **StaticGestureNet (MLP + Cutout)** | **99.95%** | **< 3 ms** | **1.37 MB** |

---

### Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python environment and dependency management.

#### 1. Prerequisites
- Python `>=3.12`
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)
- macOS (Apple Silicon / Intel), Linux, or Windows with webcam support

#### 2. Environment & Dependency Installation
```bash
# Clone the repository (if not already local)
cd /path/to/ISL

# Install all locked dependencies into virtual environment
uv sync
```

#### 3. Caching Policy
- **Model Task Cache**: The MediaPipe Hand Landmarker model (`hand_landmarker.task`, ~7.8 MB) is automatically downloaded and preserved inside `.cache/hand_landmarker.task`.
- **Preprocessed Data**: Feature extractions are cached inside `./landmark_data/` (`features.npy`, `labels.npy`, `class_names.npy`).

#### 4. Dataset Directory Structure
Ensure raw ISL images are organized under `./Indian/` with 35 class subdirectories:
```
Indian/
  ├── 1/  ├── 2/  ├── 3/  ├── 4/  ├── 5/  ├── 6/  ├── 7/  ├── 8/  ├── 9/
  ├── A/  ├── B/  ├── C/  ├── D/  ├── E/  ├── F/  ├── G/  ├── H/  ├── I/
  ├── J/  ├── K/  ├── L/  ├── M/  ├── N/  ├── O/  ├── P/  ├── Q/  ├── R/
  ├── S/  ├── T/  ├── U/  ├── V/  ├── W/  ├── X/  ├── Y/  └── Z/
```

---

### Running

#### Recommended Workflow: Model 3 (3D Landmark + Bone Geometry)

```bash
# Step 1: Preprocess dataset into 3D landmark + bone feature representations (316 dims)
uv run python collect_data.py --data_dir ./Indian --output ./landmark_data

# Step 2: Train StaticGestureNet with Landmark Cutout augmentation
uv run python model_best.py --mode train --data_dir ./landmark_data

# Step 3: Launch the interactive Gradio WebUI (Port 7862)
uv run python model_best.py --mode ui
# → Open in browser: http://127.0.0.1:7862

# Step 4: Launch Real-Time Live Webcam HUD with skeletal tracking
uv run python model_best.py --mode live
# → Press 'q' inside the OpenCV window to exit
```

#### Running Legacy Image-Based Models

```bash
# Model 1 — UltraLight (MobileNetV2)
uv run python model1_ultralight.py --mode train --data_dir ./Indian
uv run python model1_ultralight.py --mode ui
# → Open in browser: http://127.0.0.1:7860

# Model 2 — HighPerformance (ResNet50)
uv run python model2_highperf.py --mode train --data_dir ./Indian
uv run python model2_highperf.py --mode ui
# → Open in browser: http://127.0.0.1:7861
```

#### Smoke Test / Verification

```bash
# Verify preprocessed dataset integrity
uv run python -c "import numpy as np; f=np.load('landmark_data/features.npy'); print('Features shape:', f.shape)"

# Verify trained model checkpoint
uv run python -c "import torch; ckpt=torch.load('best_isl_landmark_model.pth', map_location='cpu'); print('Checkpoint classes:', len(ckpt['class_names']))"
```

---

### Documentation

#### 1. Architecture Overview: 3D Dual-Hand Feature Engineering

Instead of relying on lighting- and background-sensitive 2D image pixels ($224 \times 224 \times 3 = 150,528$ dimensions), Model 3 extracts a compact, rotation/scale-normalized **316-dimensional** vector representing physical hand geometry across both hands:

```
Dual-Hand Feature Vector (316 dimensions)
├── Hand 1 (158 dimensions)
│   ├── [0:63]    Normalized 3D Coordinates (21 landmarks × x, y, z) centered on wrist
│   ├── [63:123]  3D Bone Direction Vectors (20 segments × dx, dy, dz)
│   ├── [123:143] 3D Euclidean Bone Lengths (20 segments)
│   └── [143:158] 3D Inter-Joint Angles via Dot Product (15 joint triplets)
└── Hand 2 (158 dimensions)
    └── Identical 158-dim layout (zero-padded if only 1 hand is visible)
```

#### 2. Landmark Cutout Augmentation

Standard Cutout removes pixel patches. In structural landmark space, `AugmentedDataset` introduces **Landmark Cutout**:
- Dynamically selects 1–3 random finger sub-graphs (thumb, index, middle, ring, pinky) during training ($p=0.30$).
- Zeros out all associated 3D coordinates, bone vectors, lengths, and angles across the selected digits.
- Injects Gaussian noise ($\sigma=0.02$) to simulate sensor jitter and physical hand occlusions.
- Forces the neural network to identify gestures using remaining visible joints.

#### 3. Neural Network Architecture (`StaticGestureNet`)

```
Input: 316-dim Feature Vector
  ↓
Linear(316 → 512) → BatchNorm1d → ReLU → Dropout(0.30)
  ↓
Linear(512 → 256) → BatchNorm1d → ReLU → Dropout(0.30)
  ↓
Linear(256 → 128) → BatchNorm1d → ReLU → Dropout(0.20)
  ↓
Linear(128 →  64) → BatchNorm1d → ReLU → Dropout(0.15)
  ↓
Linear( 64 →  35) → Softmax (35 Classes)
```

#### 4. Internal Project Documentation Links

Comprehensive internal documentation is maintained inside the [`./agent/`](file:///Volumes/PORTABLESSD/Code/ISL/agent/) directory:

- 📖 **[Agent Documentation Index (`agent/README.md`)](file:///Volumes/PORTABLESSD/Code/ISL/agent/README.md)**: Main entry point for internal project documentation.
- 🏗️ **[Project Structure & Architecture (`agent/project_structure.md`)](file:///Volumes/PORTABLESSD/Code/ISL/agent/project_structure.md)**: Component mappings, directory layouts, and data flow pipelines.
- 📐 **[Logic Implementation & Mathematics (`agent/logic_implementation.md`)](file:///Volumes/PORTABLESSD/Code/ISL/agent/logic_implementation.md)**: Mathematical proofs, coordinate normalizations, angle computations, and model architectures.
- 🧪 **[Trial History & Decision Records (`agent/trial_history.md`)](file:///Volumes/PORTABLESSD/Code/ISL/agent/trial_history.md)**: Progression of experiments, failure modes, design pivots, and comparative benchmarks.
- 📋 **[Established Facts & Constraints (`agent/established_facts.md`)](file:///Volumes/PORTABLESSD/Code/ISL/agent/established_facts.md)**: System specs, cache policies, hardware acceleration flags, and port standards.

---

## License

This project is licensed under the MIT License.
