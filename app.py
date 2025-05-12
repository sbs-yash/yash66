from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import uvicorn

app = FastAPI()

# Load trained YOLOv8 model
model = YOLO("yolov8m_smoke_trained.pt")  # Load your trained model


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = np.array(img)

        # Perform inference
        results = model(img, imgsz=640)

        # Draw bounding boxes on the image
        detections = results[0].boxes  # Get detections
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = round(float(box.conf[0].item()), 2)  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = model.names[cls] if cls < len(model.names) else f"class_{cls}"
            label_text = f"{label} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert image back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_bytes = BytesIO()
        img_pil.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        return {
            "message": "Detection complete",
            "detections": [
                {
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": conf,
                    "label": label
                } for box in detections
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
