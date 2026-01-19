import torch
import os
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ------------------------------------
# PATHS  ← EDITED HERE
# ------------------------------------
IMAGE_FOLDER = r"C:\Users\user\fruit-vision\data\fruit_test2"
MODEL_PATH   = r"C:\Users\user\fruit-vision\models\mobilenet_fruit.pth"

# ------------------------------------
# DEVICE
# ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------
# MODEL SETUP
# ------------------------------------
num_classes = 12

weights = MobileNet_V2_Weights.DEFAULT
transform = weights.transforms()

model = mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

# safer load (no warning)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model = model.to(device)
model.eval()

class_names = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum',
    'fresh_orange', 'fresh_tomato',
    'stale_apple', 'stale_banana', 'stale_bitter_gourd', 'stale_capsicum',
    'stale_orange', 'stale_tomato'
]

print("\n🚀 INFERENCE ON NEW IMAGES\n")

fresh = 0
stale = 0

# ------------------------------------
# LOOP THROUGH NEW FOLDER
# ------------------------------------
for img_name in os.listdir(IMAGE_FOLDER):

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    try:
        image = Image.open(img_path).convert("RGB")
    except:
        continue

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)

    label = class_names[pred.item()]
    confidence = conf.item() * 100

    print(f"Image: {img_name}")
    print(f" → {label}")
    print(f" → Confidence: {confidence:.2f}%\n")

    if label.startswith("fresh"):
        fresh += 1
    else:
        stale += 1


print("📊 FINAL COUNT")
print("Fresh:", fresh)
print("Stale:", stale)
