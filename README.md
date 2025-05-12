# YOLOv8 Stock Market Pattern Detection

## Overview
This project utilizes the YOLOv8 model to detect stock market patterns from financial charts. The model is trained to identify key technical analysis patterns that traders and analysts use to make informed decisions.

## Features
- **Real-time stock market pattern detection**
- **Trained on financial chart images**
- **Uses YOLOv8 for high-accuracy predictions**
- **Fast inference with OpenCV visualization**

## Installation
Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install ultralytics opencv-python numpy requests pillow
```

## Usage
Below is a simple script to load the trained YOLOv8 model and perform inference on stock market charts:

```python
import os
import cv2
import requests
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# ✅ Define model path
model_path = "models/Stock_prediction/stock_market.pt"

# ✅ Load YOLO model
model = YOLO(model_path)

# ✅ Set model parameters
model.overrides['conf'] = 0.25  # Confidence threshold
model.overrides['iou'] = 0.45  # IoU threshold
model.overrides['agnostic_nms'] = False  # Class-agnostic NMS
model.overrides['max_det'] = 1000  # Max detections per image

# ✅ Provide image URL
IMG_URL = "https://www.investopedia.com/thmb/xBXLWUjf9xBev6FF0A7bie-j6J8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/dotdash_Final_Introductio_to_Technical_Analysis_Price_Patterns_Sep_2020-01-c68c49b8f38741a6b909ecc71e41f6eb.jpg"

# ✅ Fetch image from URL
response = requests.get(IMG_URL)
image = Image.open(BytesIO(response.content))
image = np.array(image)

# ✅ Convert RGB to BGR (for OpenCV)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# ✅ Run YOLO inference
results = model(image)

# ✅ Annotate results
annotated_image = results[0].plot()

# ✅ Show image in PyCharm (not Colab)
cv2.imshow("YOLO Prediction", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Model Information
- **Model Name:** [Stock Market Pattern Detection YOLOv8](https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8)
- **Architecture:** YOLOv8m
- **Dataset:** Trained on financial chart patterns dataset

## Deployment
Deploy the model using FastAPI:

```bash
pip install fastapi uvicorn
```

Create `app.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict(text: str):
    return {"prediction": "Stock pattern detected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the server:

```bash
uvicorn app:app --reload
```

## References
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [Hugging Face Model](https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8)

