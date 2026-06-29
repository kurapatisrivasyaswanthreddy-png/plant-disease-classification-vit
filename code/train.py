# =========================
# 1. Mount Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 2. Base Path (YOUR PATH)
# =========================
import os

base_path = "/content/drive/MyDrive/Plants_2"

folders = os.listdir(base_path)
print("Folders:", folders)

train_path = base_path + "/train"

# Auto detect validation
if "valid" in folders:
    val_path = base_path + "/valid"
elif "val" in folders:
    val_path = base_path + "/val"
elif "validation" in folders:
    val_path = base_path + "/validation"
else:
    print("⚠️ No validation folder → using train split")
    val_path = train_path

# Auto detect test
if "test" in folders:
    test_path = base_path + "/test"
else:
    print("⚠️ No test folder → using validation")
    test_path = val_path

# =========================
# 3. Imports
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# 4. GPU Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

torch.backends.cudnn.benchmark = True

# =========================
# 5. Model (FAST + VALID)
# =========================
MODEL_NAME = "facebook/deit-small-patch16-224"

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# =========================
# 6. Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# 7. Dataset
# =========================
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_path, transform=val_transform)
test_dataset  = datasets.ImageFolder(test_path, transform=val_transform)

print("Classes:", train_dataset.classes)

# =========================
# 8. Loaders (FAST)
# =========================
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)

# =========================
# 9. Model
# =========================
num_classes = len(train_dataset.classes)

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

model.to(device)

# =========================
# 10. Training Setup
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scaler = torch.cuda.amp.GradScaler()

# =========================
# 11. Training
# =========================
epochs = 10
best_acc = 0

train_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images = processor(images=images, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ===== Validation =====
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = processor(images=images, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)

            outputs = model(pixel_values=images).logits
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "/content/drive/MyDrive/best_model.pth")
        print("🔥 Best model saved!")

# =========================
# 12. Graphs
# =========================
plt.plot(train_losses)
plt.title("Training Loss")
plt.show()

plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.show()

# =========================
# 13. Testing
# =========================
model.eval()
correct, total = 0, 0

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        images = processor(images=images, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)

        outputs = model(pixel_values=images).logits
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Test Accuracy:", 100 * correct / total)

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes).plot()
plt.show()
# =========================
# SAVE EVERYTHING
# =========================
import json
import matplotlib.pyplot as plt

save_path = "/content/drive/MyDrive/vit_results"
os.makedirs(save_path, exist_ok=True)

# 1. Save Final Model
torch.save(model.state_dict(), save_path + "/final_model.pth")

# 2. Save Best Model (already saved during training, but safe again)
torch.save(model.state_dict(), save_path + "/best_model.pth")

# 3. Save Metrics (loss + accuracy)
results = {
    "train_loss": train_losses,
    "val_accuracy": val_accuracies,
    "test_accuracy": float(100 * correct / total)
}

with open(save_path + "/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Metrics saved")

# 4. Save Training Loss Graph
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(save_path + "/training_loss.png")
plt.close()

# 5. Save Validation Accuracy Graph
plt.figure()
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(save_path + "/validation_accuracy.png")
plt.close()

print("✅ Graphs saved")

# 6. Save Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure()
ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes).plot()
plt.title("Confusion Matrix")
plt.savefig(save_path + "/confusion_matrix.png")
plt.close()

print("✅ Confusion matrix saved")

# 7. Save Class Names
with open(save_path + "/classes.txt", "w") as f:
    for cls in train_dataset.classes:
        f.write(cls + "\n")

print("✅ Classes saved")

# =========================
print("\n🎉 ALL FILES SAVED IN DRIVE:", save_path)