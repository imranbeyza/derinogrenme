import os
import copy
import random
import warnings
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

import timm

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
BASE_PATH = r"C:\Users\Acer\retinalens-ai-diabetic-retinopathy-detection\archive (6)"
SAVE_PATH = "best_model.pth"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
NUM_CLASSES = 5
SEED = 42
NUM_WORKERS = 0  
PATIENCE = 7     

CLASS_NAMES = ['Healthy', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferate DR']


# =========================================================
# SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# TRANSFORMS
# =========================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =========================================================
# DATASET
# =========================================================
full_dataset = datasets.ImageFolder(BASE_PATH, transform=train_transforms)
print("Detected classes:", full_dataset.classes)

if len(full_dataset.classes) == NUM_CLASSES:
    CLASS_NAMES = full_dataset.classes

train_size = int((1 - VAL_SPLIT) * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)


val_dataset.dataset = copy.deepcopy(full_dataset)
val_dataset.dataset.transform = val_transforms

print(f"Total images : {len(full_dataset)}")
print(f"Train images : {len(train_dataset)}")
print(f"Val images   : {len(val_dataset)}")


# =========================================================
# SAMPLER FOR IMBALANCED DATA
# =========================================================
train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = Counter(train_targets)
print("Train class counts:", class_counts)

class_weights_list = []
for i in range(NUM_CLASSES):
    count = class_counts.get(i, 1)
    class_weights_list.append(1.0 / count)

sample_weights = [class_weights_list[label] for label in train_targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# =========================================================
# FOCAL LOSS
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =========================================================
# MODEL
# =========================================================
model = timm.create_model("efficientnet_b0", pretrained=True)

for param in model.parameters():
    param.requires_grad = False


for param in model.blocks[-3:].parameters():
    param.requires_grad = True

for param in model.conv_head.parameters():
    param.requires_grad = True

for param in model.bn2.parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(model.classifier.in_features, NUM_CLASSES)
)

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

alpha_weights = torch.tensor(
    [1.0 / class_counts.get(i, 1) for i in range(NUM_CLASSES)],
    dtype=torch.float32
).to(device)

criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    verbose=True
)


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(y_true, y_pred):
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1


# =========================================================
# TRAIN ONE EPOCH
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader)
    acc, prec, rec, f1 = calculate_metrics(all_labels, all_preds)

    return epoch_loss, acc, prec, rec, f1


# =========================================================
# EVALUATE
# =========================================================
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    acc, prec, rec, f1 = calculate_metrics(all_labels, all_preds)

    if return_preds:
        return epoch_loss, acc, prec, rec, f1, np.array(all_labels), np.array(all_preds), np.array(all_probs)

    return epoch_loss, acc, prec, rec, f1


# =========================================================
# TRAIN LOOP + EARLY STOPPING
# =========================================================
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_precision": [],
    "val_precision": [],
    "train_recall": [],
    "val_recall": [],
    "train_f1": [],
    "val_f1": []
}

best_f1 = 0.0
best_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())
patience_counter = 0

print("\n========== TRAINING STARTED ==========")

for epoch in range(EPOCHS):
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
        model, val_loader, criterion, device
    )

    scheduler.step(val_f1)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_precision"].append(train_prec)
    history["val_precision"].append(val_prec)
    history["train_recall"].append(train_rec)
    history["val_recall"].append(val_rec)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"LR          : {current_lr:.8f}")
    print(f"Train Loss  : {train_loss:.4f}")
    print(f"Val Loss    : {val_loss:.4f}")
    print(f"Train Acc   : {train_acc:.4f} | Val Acc   : {val_acc:.4f}")
    print(f"Train Prec  : {train_prec:.4f} | Val Prec  : {val_prec:.4f}")
    print(f"Train Recall: {train_rec:.4f} | Val Recall: {val_rec:.4f}")
    print(f"Train F1    : {train_f1:.4f} | Val F1    : {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), SAVE_PATH)
        patience_counter = 0
        print("Best model saved!")
    else:
        patience_counter += 1
        print(f"No improvement. Early stop counter: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

print("\n========== TRAINING FINISHED ==========")
print(f"Best epoch: {best_epoch}")
print(f"Best val F1: {best_f1:.4f}")


model.load_state_dict(best_model_wts)


# =========================================================
# FINAL EVALUATION
# =========================================================
val_loss, val_acc, val_prec, val_rec, val_f1, y_true, y_pred, y_probs = evaluate(
    model, val_loader, criterion, device, return_preds=True
)

cm = confusion_matrix(y_true, y_pred)

print("\n========== FINAL EVALUATION ==========")
print(f"Loss      : {val_loss:.4f}")
print(f"Accuracy  : {val_acc:.4f}")
print(f"Precision : {val_prec:.4f}")
print(f"Recall    : {val_rec:.4f}")
print(f"F1 Score  : {val_f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


# =========================================================
# PLOTS
# =========================================================
def plot_training_curves(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], marker='o', label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], marker='o', label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_acc"], marker='o', label="Train Accuracy")
    plt.plot(epochs_range, history["val_acc"], marker='o', label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Precision
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_precision"], marker='o', label="Train Precision")
    plt.plot(epochs_range, history["val_precision"], marker='o', label="Val Precision")
    plt.title("Precision Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Recall
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_recall"], marker='o', label="Train Recall")
    plt.plot(epochs_range, history["val_recall"], marker='o', label="Val Recall")
    plt.title("Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # F1
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_f1"], marker='o', label="Train F1")
    plt.plot(epochs_range, history["val_f1"], marker='o', label="Val F1")
    plt.title("F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cm, class_names):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Final Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts, class_names):
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts)
    plt.title("Training Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


plot_training_curves(history)
plot_confusion_matrices(cm, CLASS_NAMES)
plot_class_distribution(class_counts, CLASS_NAMES)


# =========================================================
# SHOW SAMPLE PREDICTIONS
# =========================================================
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor.cpu() * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor.permute(1, 2, 0).numpy()


def show_sample_predictions(model, loader, class_names, num_images=9):
    model.eval()
    images_shown = 0

    plt.figure(figsize=(14, 10))
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    plt.show()
                    return

                plt.subplot(3, 3, images_shown + 1)
                img = denormalize(images[i])

                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]
                conf = probs[i][preds[i]].item() * 100

                plt.imshow(img)
                plt.title(f"T: {true_label}\nP: {pred_label} ({conf:.1f}%)", fontsize=9)
                plt.axis("off")
                images_shown += 1

    plt.tight_layout()
    plt.show()


show_sample_predictions(model, val_loader, CLASS_NAMES, num_images=9)


# =========================================================
# GRAD-CAM
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients[0]      # [C,H,W]
        activations = self.activations[0]  # [C,H,W]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def preprocess_image_for_gradcam(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")

    image_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = val_transforms
    pil_like = image_rgb
    tensor = transform(pil_like).unsqueeze(0).to(device)

    return image_rgb, tensor


def show_gradcam(image_path, model, class_names):
    image_rgb, input_tensor = preprocess_image_for_gradcam(image_path)

    
    target_layer = model.blocks[-1]
    gradcam = GradCAM(model, target_layer)

    cam, pred_idx = gradcam.generate(input_tensor)
    gradcam.remove_hooks()

    heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf = probs[0][pred_idx].item() * 100

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Prediction: {class_names[pred_idx]} ({conf:.2f}%)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# OPTIONAL: TEK GÖRSELDE TEST
# =========================================================
def predict_single_image(image_path, model, class_names):
    model.eval()
    image_rgb, input_tensor = preprocess_image_for_gradcam(image_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    print("\nSingle Image Prediction")
    print("Predicted class:", class_names[pred_idx])
    print("Confidence: %.2f%%" % (probs[pred_idx].item() * 100))
    print("Probabilities:")
    for i, c in enumerate(class_names):
        print(f"  {c}: {probs[i].item() * 100:.2f}%")

    plt.figure(figsize=(5, 5))
    plt.imshow(image_rgb)
    plt.title(f"{class_names[pred_idx]} ({probs[pred_idx].item() * 100:.2f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


