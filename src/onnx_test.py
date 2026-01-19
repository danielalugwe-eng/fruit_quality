import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession(r"C:\Users\user\fruit-vision\models\mobilenet_fruit.onnx")

# Example: create a dummy input (1, 3, 224, 224)
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
print("ONNX inference output shape:", outputs[0].shape)
