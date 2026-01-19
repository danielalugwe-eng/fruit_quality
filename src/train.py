import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
import os

# =========================================
# PATHS
# =========================================
train_path = r"C:\Users\user\fruit-vision\data\train"
val_path   = r"C:\Users\user\fruit-vision\data\val"

model_save_path = r"C:\Users\user\fruit-vision\models\mobilenet_fruit.pth"

# =========================================
# HYPERPARAMETERS
# =========================================
batch_size = 16
num_epochs = 10
learning_rate = 1e-3
num_classes = 12   # fresh + stale fruits

# =========================================
# DEVICE
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================
# MODEL (Transfer Learning)
# =========================================
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Replace classifier head
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# =========================================
# DATASETS
# =========================================
transform = weights.transforms()

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset   = datasets.ImageFolder(root=val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", train_dataset.class_to_idx)

# =========================================
# LOSS & OPTIMIZER
# =========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =========================================
# TRAINING LOOP
# =========================================
print("Starting training...\n")

for epoch in range(num_epochs):

    # -------- TRAIN PHASE --------
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # -------- VALIDATION PHASE --------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Val Accuracy:   {val_acc:.2f}%\n")


# =========================================
# SAVE MODEL
# =========================================
torch.save(model.state_dict(), model_save_path)
print("Training complete.")
print("Model saved to:", model_save_path)
