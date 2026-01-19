import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Absolute path to your dataset folder
dataset_path = r"C:\Users\user\fruit-vision\data\fruit_dataset"

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Print class mapping
print("Class mapping:", dataset.class_to_idx)

# DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Test one batch
for images, labels in loader:
    print(images.shape, labels)
    break
