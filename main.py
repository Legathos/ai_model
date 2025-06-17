from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import base64
import io
import numpy as np
import uvicorn
from pathlib import Path
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI(title="Food Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Define request model for base64 image
class ImageRequest(BaseModel):
    image_base64: str  # Base64 encoded image

# Load model and processor
model_name = "prithivMLmods/Food-101-93M"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Food-101 labels
labels = {
    "0": "apple_pie", "1": "baby_back_ribs", "2": "baklava", "3": "beef_carpaccio", "4": "beef_tartare",
    "5": "beet_salad", "6": "beignets", "7": "bibimbap", "8": "bread_pudding", "9": "breakfast_burrito",
    "10": "bruschetta", "11": "caesar_salad", "12": "cannoli", "13": "caprese_salad", "14": "carrot_cake",
    "15": "ceviche", "16": "cheesecake", "17": "cheese_plate", "18": "chicken_curry", "19": "chicken_quesadilla",
    "20": "chicken_wings", "21": "chocolate_cake", "22": "chocolate_mousse", "23": "churros", "24": "clam_chowder",
    "25": "club_sandwich", "26": "crab_cakes", "27": "creme_brulee", "28": "croque_madame", "29": "cup_cakes",
    "30": "deviled_eggs", "31": "donuts", "32": "dumplings", "33": "edamame", "34": "eggs_benedict",
    "35": "escargots", "36": "falafel", "37": "filet_mignon", "38": "fish_and_chips", "39": "foie_gras",
    "40": "french_fries", "41": "french_onion_soup", "42": "french_toast", "43": "fried_calamari", "44": "fried_rice",
    "45": "frozen_yogurt", "46": "garlic_bread", "47": "gnocchi", "48": "greek_salad", "49": "grilled_cheese_sandwich",
    "50": "grilled_salmon", "51": "guacamole", "52": "gyoza", "53": "hamburger", "54": "hot_and_sour_soup",
    "55": "hot_dog", "56": "huevos_rancheros", "57": "hummus", "58": "ice_cream", "59": "lasagna",
    "60": "lobster_bisque", "61": "lobster_roll_sandwich", "62": "macaroni_and_cheese", "63": "macarons",
    "64": "miso_soup",
    "65": "mussels", "66": "nachos", "67": "omelette", "68": "onion_rings", "69": "oysters",
    "70": "pad_thai", "71": "paella", "72": "pancakes", "73": "panna_cotta", "74": "peking_duck",
    "75": "pho", "76": "pizza", "77": "pork_chop", "78": "poutine", "79": "prime_rib",
    "80": "pulled_pork_sandwich", "81": "ramen", "82": "ravioli", "83": "red_velvet_cake", "84": "risotto",
    "85": "samosa", "86": "sashimi", "87": "scallops", "88": "seaweed_salad", "89": "shrimp_and_grits",
    "90": "spaghetti_bolognese", "91": "spaghetti_carbonara", "92": "spring_rolls", "93": "steak",
    "94": "strawberry_shortcake",
    "95": "sushi", "96": "tacos", "97": "takoyaki", "98": "tiramisu", "99": "tuna_tartare", "100": "waffles"
}


def classify_food(image):
    """Predicts the type of food in the image."""
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    # Sort by descending probability and get top 5
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:5])

    return predictions


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/predict')
@app.post('/estimateFoodFromImage')
async def predict(request: ImageRequest):
    try:
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Decode base64 image
        try:
            # Remove potential data URL prefix
            if ',' in request.image_base64:
                base64_data = request.image_base64.split(',')[1]
            else:
                base64_data = request.image_base64

            # Decode base64 string to bytes
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            predictions = classify_food(image)

            return {"predictions": predictions}
        except base64.binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
