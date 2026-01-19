import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn
import onnx

# Full path to your trained PyTorch model
MODEL_PATH = r"C:\Users\user\fruit-vision\models\mobilenet_fruit.pth"
ONNX_PATH = r"C:\Users\user\fruit-vision\models\mobilenet_fruit.onnx"

# Load model
num_classes = 12
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Dummy input to export (batch_size=1, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

print(f"ONNX model exported to: {ONNX_PATH}")
