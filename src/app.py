import os
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# ----------------------------
# FASTAPI APP
# ----------------------------
app = FastAPI(title="Fruit Vision API")

# ----------------------------
# CONFIG
# ----------------------------

# Path works both in Docker and local
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = "models/mobilenet_fruit.pth"

NUM_CLASSES = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD MODEL
# ----------------------------

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Safe load
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

# ----------------------------
# CLASS NAMES (MUST MATCH TRAIN)
# ----------------------------

class_names = [
    'fresh_apple','fresh_banana','fresh_bitter_gourd','fresh_capsicum',
    'fresh_orange','fresh_tomato',
    'stale_apple','stale_banana','stale_bitter_gourd','stale_capsicum',
    'stale_orange','stale_tomato'
]

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----------------------------
# HEALTH CHECK
# ----------------------------

@app.get("/")
def home():
    return {
        "status": "ok",
        "model": "mobilenet_v2",
        "classes": NUM_CLASSES
    }

# ----------------------------
# PREDICT ENDPOINT
# ----------------------------

@app.post("/predict")
async def predict(file: UploadFile):

    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = class_names[pred_idx.item()]
    confidence = round(conf.item() * 100, 2)

    return {
        "class": label,
        "confidence": confidence
    }
