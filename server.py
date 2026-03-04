
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
CLASS_NAMES = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Cherry_(including_sour)___healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)__Common_rust",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"
]

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
