from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load trained model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model.predict(file_path)

    probs = results[0].probs

    class_id = probs.top1
    confidence = float(probs.top1conf)

    label = model.names[class_id]

    # Structural Health Calculation
    if label == "Negative":
        health = confidence * 100
    else:
        health = (1 - confidence) * 100

    # Health Status
    if health > 80:
        status = "Healthy"
    elif health > 50:
        status = "Moderate Damage"
    else:
        status = "Severe Damage"

    return JSONResponse({
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "structural_health": round(health, 2),
        "status": status
    })