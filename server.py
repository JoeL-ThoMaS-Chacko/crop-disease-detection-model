
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import base64, io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = tf.keras.models.load_model("plant_disease_model.keras")
CLASS_NAMES = ["Healthy", "Leaf Blight", "Powdery Mildew"]  

class ImageRequest(BaseModel):
    image: str  

@app.post("/predict")
async def predict(req: ImageRequest):
    img_bytes = base64.b64decode(req.image)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    return { "disease": CLASS_NAMES[idx], "confidence": round(float(preds[0][idx]) * 100, 2) }