# Established Facts & Operational Constraints

## Overview

This document catalogs all verified facts, hardware specifications, environmental configurations, dataset characteristics, and networking/port standards established across the ISL project.

---

## 1. Dataset & Class Configuration

- **Dataset Path**: `./Indian`
- **Total Classes**: 35 classes
  - Numeric Digits (9): `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
  - Alphabetic Letters (26): `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
- **Total Samples Preprocessed**: 42,364 images across 35 classes.
- **Preprocessed Array Artifacts**:
  - `landmark_data/features.npy`: Shape `(42364, 316)`, data type `float32`, file size ~53.5 MB.
  - `landmark_data/labels.npy`: Shape `(42364,)`, data type `int64`, file size ~339 KB.
  - `landmark_data/class_names.npy`: Shape `(35,)`, containing sorted class labels `['1', '2', ..., '9', 'A', ..., 'Z']`.

---

## 2. Checkpoint & Artifact Specs

- **Trained Model Weights**: `best_isl_landmark_model.pth`
- **File Size**: ~1.37 MB (1,374,409 bytes)
- **Checkpoint Keys**:
  - `model_state`: PyTorch `state_dict` mapping parameters for `StaticGestureNet`.
  - `num_classes`: Integer `35`.
  - `input_dim`: Integer `316`.
  - `class_names`: Python list of 35 strings `['1', '2', ..., 'Z']`.
- **Validation Accuracy**: 99.95% on held-out validation set.

---

## 3. MediaPipe & Caching Policy

- **Model Bundle**: `hand_landmarker.task` (MediaPipe Tasks API Hand Landmarker float16 model).
- **Download URL**: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- **Cache Location**: `./.cache/hand_landmarker.task` (size: ~7.8 MB).
- **Detection Behavior**:
  - The detector checks for legacy `mediapipe.solutions.hands.Hands`. If unavailable (e.g. newer MediaPipe releases or headless environments), it automatically falls back to `HandLandmarker.create_from_options` using `./.cache/hand_landmarker.task`.
  - Ensures fully offline execution once downloaded to `.cache/`.

---

## 4. Hardware Acceleration & Device Dispatch

- **Device Dispatch Logic (`get_device()` in all scripts)**:
  1. Checks `torch.cuda.is_available()` $\to$ `cuda` (NVIDIA GPUs).
  2. Checks `torch.backends.mps.is_available()` $\to$ `mps` (Apple Silicon Metal Performance Shaders on macOS).
  3. Fallback $\to$ `cpu`.
- **Memory Footprint**:
  - `StaticGestureNet`: ~237K parameters, peak memory < 50MB RAM during live inference.
  - CPU Inference Latency: < 3 ms per sample.
  - MediaPipe Landmark Detection: ~15-25 ms per frame on CPU / Apple Silicon.

---

## 5. WebUI Port Allocations

To prevent port collisions between simultaneous models, distinct ports are assigned:

| Application / Script | Default Port | URL | Framework |
| :--- | :--- | :--- | :--- |
| **`model1_ultralight.py --mode ui`** | `7860` | `http://127.0.0.1:7860` | Gradio Interface |
| **`model2_highperf.py --mode ui`** | `7861` | `http://127.0.0.1:7861` | Gradio Interface |
| **`model_best.py --mode ui`** | `7862` | `http://127.0.0.1:7862` | Gradio Interface |
| **`model_best.py --mode live`** | N/A (Window) | OpenCV Native Window | OpenCV HighGUI |

---

## 6. Dependency & Package Management Rules

- **Package Manager**: All Python scripts and operations MUST run via `uv` (e.g., `uv run python <script>.py`).
- **Python Version**: `>=3.12` defined in `pyproject.toml` and `.python-version`.
- **Operating Environment**: Tested on macOS (Darwin arm64) and Linux x86_64.
- **Cache Rule**: Local cache directories (`.cache/`, `landmark_data/`) are kept inside the codebase and preserved for reproducibility.
