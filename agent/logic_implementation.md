# Logic Implementation & Mathematical Formulation

## Overview

This document provides an exhaustive breakdown of the algorithms, feature engineering mathematics, augmentation strategies, neural network architectures, and runtime loops implemented across the ISL codebase.

---

## 1. 3D Geometric Feature Extraction

Rather than feeding raw pixels (e.g. 224x224x3 = 150,528 dimensions) directly into heavy convolutional networks, the system extracts a compressed, lighting- and background-invariant 3D structural hand geometry vector.

### Hand Landmark Representation
MediaPipe extracts 21 keypoints per hand in 3D space:
$$L_i = (x_i, y_i, z_i) \quad \text{for } i \in \{0, 1, \dots, 20\}$$
where $L_0$ is the wrist landmark.

### Step 1: Wrist Centering & Scale Normalization
To ensure scale and translation invariance:
1. **Centering**:
   $$\tilde{L}_i = L_i - L_0 \quad \forall i \in \{0, \dots, 20\}$$
2. **Scale Factor**:
   $$s = \max\left(\max_{i} \|\tilde{L}_i\|_2, 10^{-6}\right)$$
3. **Normalized Coordinates**:
   $$\hat{L}_i = \frac{\tilde{L}_i}{s}$$
Flattened 3D coordinates yield $21 \times 3 = 63$ dimensions.

### Step 2: 3D Bone Direction Vectors & Lengths
The hand skeleton consists of 20 directional bone segments connecting the 21 landmarks:
- **Thumb**: $(0,1), (1,2), (2,3), (3,4)$
- **Index**: $(0,5), (5,6), (6,7), (7,8)$
- **Middle**: $(0,9), (9,10), (10,11), (11,12)$
- **Ring**: $(0,13), (13,14), (14,15), (15,16)$
- **Pinky**: $(0,17), (17,18), (18,19), (19,20)$

For each bone segment $k = (a, b)$:
1. **3D Bone Direction Vector**:
   $$\vec{v}_k = \hat{L}_b - \hat{L}_a = (\Delta x_k, \Delta y_k, \Delta z_k) \quad \implies 20 \times 3 = 60 \text{ dims}$$
2. **3D Bone Length (Euclidean)**:
   $$d_k = \|\vec{v}_k\|_2 = \sqrt{\Delta x_k^2 + \Delta y_k^2 + \Delta z_k^2} \quad \implies 20 \text{ dims}$$

### Step 3: Inter-Joint 3D Angles
To capture finger curvature, flexure, and spread, 15 joint angle triplets $(a, b, c)$ are defined across the 5 digits:
- **Thumb**: $(0,1,2), (1,2,3), (2,3,4)$
- **Index**: $(0,5,6), (5,6,7), (6,7,8)$
- **Middle**: $(0,9,10), (9,10,11), (10,11,12)$
- **Ring**: $(0,13,14), (13,14,15), (14,15,16)$
- **Pinky**: $(0,17,18), (17,18,19), (18,19,20)$

For each joint triplet, the angle $\theta_j$ is computed using the 3D dot product:
$$\vec{u} = \hat{L}_a - \hat{L}_b, \quad \vec{w} = \hat{L}_c - \hat{L}_b$$
$$\cos(\theta_j) = \text{clip}\left(\frac{\vec{u} \cdot \vec{w}}{\|\vec{u}\|_2 \|\vec{w}\|_2 + 10^{-8}}, -1.0, 1.0\right)$$
$$\theta_j = \arccos(\cos(\theta_j)) \quad \implies 15 \text{ dims}$$

### Single-Hand Vector Summary
$$\text{Single Hand Feature Vector} = [\hat{L}_{0..20} \,(63) \mid \vec{v}_{0..19} \,(60) \mid d_{0..19} \,(20) \mid \theta_{0..14} \,(15)] \implies \mathbf{158 \text{ dims}}$$

### Dual-Hand Concatenation
Many ISL gestures require two hands. The system extracts up to 2 hands:
$$\mathbf{X} = [\mathbf{h}_1 \in \mathbb{R}^{158}, \; \mathbf{h}_2 \in \mathbb{R}^{158}] \implies \mathbf{X} \in \mathbb{R}^{316}$$
If only 1 hand is present in an image, $\mathbf{h}_2$ is zero-padded ($\mathbf{0}_{158}$).

---

## 2. Landmark Cutout Augmentation (`AugmentedDataset`)

### Motivation
Standard data augmentations (e.g. Cutout, RandomErasing) operate on 2D pixel grids. When operating on 1D structural geometry, physical occlusions (such as fingers crossing or one hand occluding the other) lead to missing or noisy landmarks.

### Implementation Logic
`AugmentedDataset` dynamically masks finger clusters during training:
1. **Trigger Condition**: Applied with probability $p = 0.30$.
2. **Finger Selection**: Randomly sample $k \in \{1, 2, 3\}$ fingers from $\{\text{thumb, index, middle, ring, pinky}\}$.
3. **Index Masking**:
   - Precomputed `FINGER_MASKS` maps each finger to its corresponding slice in the 158-dim feature vector:
     - 3D Coordinates: 4 landmarks $\times 3 = 12$ indices
     - Bone Vectors: 4 bones $\times 3 = 12$ indices
     - Bone Lengths: 4 indices
     - Joint Angles: 3 indices
   - Total indices masked per finger: $12 + 12 + 4 + 3 = 31$ indices.
4. **Multi-Hand Drop Probability**:
   - Mask Hand 1 with 70% probability.
   - Mask Hand 2 with 50% probability.
5. **Gaussian Noise Regularization**:
   $$\mathbf{X}_{\text{aug}} = \mathbf{X}_{\text{masked}} + \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \sigma = 0.02$$

```python
# Example: Masking the Index Finger (Landmarks 5-8, Bones 4-7, Angles 3-5)
# Coordinates (12 dims): [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
# Bone Vectors (12 dims): [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
# Bone Lengths (4 dims):  [127, 128, 129, 130]
# Angles (3 dims):        [146, 147, 148]
```

---

## 3. Deep Learning Architecture (`StaticGestureNet`)

The primary model is an optimized multi-layer perceptron with Batch Normalization, Dropout regularization, and progressive dimensionality reduction:

```
Input Vector (316 dims)
       │
   Linear(316 → 512) ── BatchNorm1d(512) ── ReLU ── Dropout(p=0.30)
       │
   Linear(512 → 256) ── BatchNorm1d(256) ── ReLU ── Dropout(p=0.30)
       │
   Linear(256 → 128) ── BatchNorm1d(128) ── ReLU ── Dropout(p=0.20)
       │
   Linear(128 →  64) ── BatchNorm1d( 64) ── ReLU ── Dropout(p=0.15)
       │
   Linear( 64 →  35)
       │
   Softmax Output (35 Classes: Digits 1-9, Letters A-Z)
```

- **Parameter Count**: 237,475 parameters (~950 KB in memory).
- **Optimization**:
  - Loss Function: `nn.CrossEntropyLoss(label_smoothing=0.05)`
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
  - Learning Rate Scheduler: `CosineAnnealingLR(T_max=60, eta_min=1e-6)`
  - Early Stopping: `patience=15`

---

## 4. Transfer Learning Baseline Architectures

### Model 1: `UltraLightISLNet` (`model1_ultralight.py`)
- **Backbone**: MobileNetV2 (`models.mobilenet_v2(weights=DEFAULT)`)
- **Input**: $(3, 224, 224)$ normalized image
- **Head**:
  - AdaptiveAvgPool2d((1, 1)) $\to 1280$
  - Dropout(0.3) $\to$ Linear(1280, 256) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(256, 35)
- **Augmentation**: RandomCrop, RandomRotation(15°), ColorJitter, RandomErasing.

### Model 2: `HighPerformanceISLNet` (`model2_highperf.py`)
- **Backbone**: ResNet50 (`models.resnet50(weights=DEFAULT)`)
- **Layer Freezing**: Early convolutional layers (conv1 through layer2) frozen; layer3, layer4, and head fine-tuned.
- **Differential Learning Rates**:
  - Backbone parameters: $\text{LR} = 10^{-4}$
  - Classification head: $\text{LR} = 10^{-3}$
- **Class Balancing**: `WeightedRandomSampler` inversely proportional to class frequencies.
- **Head**:
  - Dropout(0.4) $\to$ Linear(2048, 512) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(512, 256) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(256, 35)

---

## 5. Runtime Inference & UI Architecture

### MediaPipe Dual-API Support Strategy
To support both legacy environments (`mp.solutions.hands`) and modern MediaPipe Vision Tasks API (`HandLandmarker`), a detection wrapper dynamically detects the available backend:
- Priority 1: Attempts legacy `mp.solutions.hands.Hands`.
- Priority 2: Automatically downloads and caches `hand_landmarker.task` to `.cache/` and instantiates `HandLandmarker`.

### Real-Time Live Webcam Loop (`--mode live`)
1. Captures webcam frame via OpenCV (`1280x720`).
2. Converts frame to RGB.
3. Extracts dual-hand 2D landmarks for visualization and 3D landmarks for inference.
4. Draws color-coded skeleton (Hand 1: Warm/Cyan palette; Hand 2: Cool/Magenta palette) with finger tips annotated.
5. Feeds 316-dim vector to `StaticGestureNet` on CPU/MPS/CUDA.
6. Renders semi-transparent HUD showing prediction, confidence percentage, and active hand count at 30+ FPS.
