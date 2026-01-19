import cv2
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

# -------------------------------
# Paths and classes
# -------------------------------
MODEL_PATH = r"C:\Users\user\fruit-vision\models\mobilenet_fruit.onnx"

classes = [
    "fresh_apple", "fresh_banana", "fresh_bitter_gourd", "fresh_capsicum", 
    "fresh_orange", "fresh_tomato", "stale_apple", "stale_banana", 
    "stale_bitter_gourd", "stale_capsicum", "stale_orange", "stale_tomato"
]

# -------------------------------
# Load ONNX model
# -------------------------------
ort_session = ort.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name
print("ONNX model loaded:", MODEL_PATH)

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img)
    return img_t.unsqueeze(0).numpy()  # (1, 3, 224, 224)

# -------------------------------
# Webcam loop
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    input_tensor = preprocess(frame)

    # Run ONNX inference
    outputs = ort_session.run(None, {input_name: input_tensor})
    preds = outputs[0]
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    # Display
    label = f"{classes[class_idx]}: {confidence*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.imshow("Fruit Detection ONNX", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
