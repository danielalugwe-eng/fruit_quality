import torch
import numpy as np
from torchvision import datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------
# PATHS
# ------------------------------------
test_path = r"C:\Users\user\fruit-vision\data\test"
model_path = r"C:\Users\user\fruit-vision\models\mobilenet_fruit.pth"

# ------------------------------------
# DEVICE
# ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------
# DATASET
# ------------------------------------
weights = MobileNet_V2_Weights.DEFAULT
transform = weights.transforms()

test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class_names = test_dataset.classes
print("Classes:", test_dataset.class_to_idx)

# ------------------------------------
# LOAD MODEL
# ------------------------------------
num_classes = 12

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded.\n")

# ------------------------------------
# EVALUATION
# ------------------------------------
all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

# ------------------------------------
# RESULTS
# ------------------------------------
accuracy = 100 * correct / total
print(f"TEST ACCURACY: {accuracy:.2f}%\n")

# Per-class report
print("CLASSIFICATION REPORT:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

print("CONFUSION MATRIX:")
print(cm)

# ------------------------------------
# Fresh vs Stale summary
# ------------------------------------
fresh = 0
stale = 0

for p in all_preds:
    name = class_names[p]
    if name.startswith("fresh"):
        fresh += 1
    else:
        stale += 1

print("\nSUMMARY:")
print("Fresh detected:", fresh)
print("Stale detected:", stale)
