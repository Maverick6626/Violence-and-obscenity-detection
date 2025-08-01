from fastapi import FastAPI, UploadFile, File
from model_utils import predict
from keyframe import extract_keyframes
import uuid, os, shutil

app = FastAPI()

@app.post("/classify")
async def classify_video(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = predict(extract_keyframes, temp_path)

    os.remove(temp_path)
    return result
