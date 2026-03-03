"""
Model 1 — UltraLight ISL Classifier
Uses MobileNetV2 (pretrained on ImageNet) as a lightweight backbone
with a custom classification head for 35 ISL classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import pathlib
import os
import gradio as gr
from PIL import Image
import time


# ── Helpers ──────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


NUM_CLASSES = 35
CLASS_INDEX_TO_CHAR = {}
for i in range(NUM_CLASSES):
    CLASS_INDEX_TO_CHAR[i] = str(i + 1) if i < 9 else chr(ord('A') + i - 9)

WEIGHTS_FILE = "ultralight_mobilenet_isl.pth"


# ── Model ────────────────────────────────────────────────────────────
class UltraLightISLNet(nn.Module):
    """MobileNetV2 backbone with a compact classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features          # keep conv layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.last_channel        # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Validation wrapper (apply clean transforms) ─────────────────────
class ValidationDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        orig = self.subset.indices[idx]
        path, label = self.subset.dataset.samples[orig]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)


# ── Training ─────────────────────────────────────────────────────────
def train_model(data_dir_path: str):
    device = get_device()
    print(f"[INIT] Device selected: {device}")

    data_dir = pathlib.Path(data_dir_path)
    if not data_dir.exists():
        print(f"[ERROR] Dataset directory '{data_dir}' not found.")
        return

    img_size = 224  # MobileNetV2 native input
    batch_size = 64
    epochs = 30
    lr = 3e-4

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("[DATA] Loading dataset …")
    def _valid(path: str) -> bool:
        """Skip macOS resource forks and non-image junk."""
        return not os.path.basename(path).startswith("._")

    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tf, is_valid_file=_valid)
    num_classes = len(full_ds.classes)
    print(f"[DATA] Found {num_classes} classes | {len(full_ds)} total images")

    train_n = int(0.85 * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    val_ds_clean = ValidationDataset(val_ds, val_tf)
    print(f"[DATA] Split → train={train_n}, val={val_n}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds_clean, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)

    print("[MODEL] Building UltraLightISLNet (MobileNetV2 backbone) …")
    model = UltraLightISLNet(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_params:,} | Trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience, wait = 7, 0
    print(f"[TRAIN] Starting training for {epochs} epochs …\n")

    for epoch in range(epochs):
        t0 = time.time()
        # ── train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| loss={loss.item():.4f} acc={100*correct/total:.1f}%")

        train_acc = 100 * correct / total

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                v_loss += criterion(out, labels).item()
                _, pred = out.max(1)
                v_total += labels.size(0)
                v_correct += pred.eq(labels).sum().item()
        val_acc = 100 * v_correct / v_total
        val_loss = v_loss / len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"[Epoch {epoch+1}/{epochs}] train_acc={train_acc:.2f}% | "
              f"val_acc={val_acc:.2f}% | val_loss={val_loss:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_FILE)
            print(f"  ✅ New best model saved ({val_acc:.2f}%)")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break

    print(f"\n[DONE] Training complete. Best val accuracy: {best_val_acc:.2f}%")


# ── Inference ────────────────────────────────────────────────────────
_cached_model = None

def _load_model(device):
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    m = UltraLightISLNet(NUM_CLASSES)
    m.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
    m.to(device).eval()
    _cached_model = m
    return m

def predict(image):
    if image is None:
        return {"Error — upload an image first": 1.0}

    device = get_device()
    try:
        model = _load_model(device)
    except FileNotFoundError:
        return {"Error — train the model first (--mode train)": 1.0}

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    tensor = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        top3_p, top3_i = probs.topk(3)
        return {CLASS_INDEX_TO_CHAR[top3_i[0][j].item()]: float(top3_p[0][j])
                for j in range(3)}


# ── Gradio UI ────────────────────────────────────────────────────────
def launch_ui():
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload ISL Sign Image"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="ISL Classification — UltraLight (MobileNetV2)",
        description="Upload an ISL character image to classify it. "
                    "Returns the top-3 predicted characters with confidence.",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860)


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ISL UltraLight Model")
    p.add_argument("--mode", choices=["train", "ui"], required=True)
    p.add_argument("--data_dir", default="./Indian", help="Dataset root")
    args = p.parse_args()

    if args.mode == "train":
        train_model(args.data_dir)
    else:
        launch_ui()
