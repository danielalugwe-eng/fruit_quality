# 🍎 Fruit Vision – Smart Fruit Classification & Monitoring

## Project Overview

**Fruit Vision** is a real-time fruit ripeness detection system using **computer vision and deep learning**. The system can classify fruits into **fresh** or **stale/rotten** categories and supports live **webcam detection**, **image batch prediction**, and **export to ONNX** for edge deployment in embedded devices or robotic systems.

**Supported fruits:**

* Apple
* Banana
* Tomato
* Bitter Gourd
* Capsicum
* Orange

**Key features:**

* Lightweight **MobileNetV2** CNN model for fast inference.
* Works with individual images or live camera feed.
* ONNX export for deployment on edge devices or integration with robotic harvesters.
* FastAPI REST API for easy integration in applications.
* Dockerized deployment for cloud or local hosting.

---

## Benefits in Agriculture & Robotics

**Agriculture Sector:**

* Automates fruit quality inspection in **processing plants**.
* Reduces manual inspection errors and labor costs.
* Provides **real-time analytics** for farm management (fresh vs stale count).

**Robotics & Harvesting:**

* Integrates with **robotic fruit pickers** to harvest only ripe fruits.
* Minimizes fruit wastage by distinguishing stale/rotten produce.
* Supports **conveyor belt sorting systems** for packing houses.

---

## Project Structure

```
fruit-vision/
├── data/                   # Dataset (train, val, test, new images)
├── models/                 # Trained model files (.pth, .onnx)
├── src/                    # Source code
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script for new images
│   ├── onnx_export.py      # Export PyTorch model to ONNX
│   ├── webcam.py           # Live webcam inference
│   ├── app.py              # FastAPI app
│   └── onnx_test.py        # Test ONNX inference
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build file
└── README.md               # This documentation
```

---

## Installation & Setup (Local Machine)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/username/fruit-vision.git
cd fruit-vision
```

### 2️⃣ Create an isolated environment (Conda recommended)

```bash
conda create -n uv python=3.10
conda activate uv
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Verify the model

Make sure the trained model exists:

```text
models/mobilenet_fruit.pth
```

You can also test ONNX:

```text
models/mobilenet_fruit.onnx
```

---

## Usage Guidelines

### 1️⃣ Predict on a single image

```bash
python src/predict.py --image "data/new_fruit_test/banana.jpg"
```

**Output example:**

```
Fruit: fresh_banana
Confidence: 99.96%
```

### 2️⃣ Batch prediction

Place multiple images in a folder:

```bash
python src/predict.py --folder "data/new_fruit_test"
```

**Output:** Fresh vs Stale summary and individual predictions.

### 3️⃣ Live Webcam Detection

```bash
python src/webcam.py
```

* A window will open with the live camera feed.
* Detected fruit type and ripeness label will appear on top right.
* Press **`q`** to quit.

### 4️⃣ FastAPI REST API

Start API server:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

**Endpoint:** `POST /predict`

* Upload an image to get JSON response:

```json
{
  "class": "fresh_banana",
  "confidence": 0.9996
}
```

### 5️⃣ Export Model to ONNX (for edge deployment)

```bash
python src/onnx_export.py
```

* Output: `models/mobilenet_fruit.onnx`
* Can be used with ONNX Runtime for fast inference in embedded devices or robotics.

---

## Deployment with Docker

1️⃣ Build Docker image:

```bash
docker build -t fruit-vision-api .
```

2️⃣ Run container:

```bash
docker run -p 8000:8000 fruit-vision-api
```

* FastAPI API available at `http://localhost:8000`.
* Use `/docs` for Swagger UI testing.

---

## Notes & Tips

* Ensure the **dataset paths** in `train.py` and `predict.py` match your local structure.
* For **robotic integration**, use the **ONNX exported model** with your device inference engine.
* Webcam labels may display a default prediction at startup; this is normal as the model always predicts on the first frame.

---

## Future Enhancements

* Add **fruit weight and size estimation** using image processing.
* Integrate with **IoT sensors** for automated harvesting data logging.
* Expand to more fruits and vegetables for commercial applications.

---

## References

* [PyTorch MobileNetV2 Documentation](https://pytorch.org/vision/stable/models.html)
* [ONNX Runtime](https://onnxruntime.ai/)
* [FastAPI](https://fastapi.tiangolo.com/)
