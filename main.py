"""A FastAPI application for sentiment analysis using a fine-tuned BERT
model on the IMDB dataset."""
import os
import logging
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Logging started.")

# Initialize FastAPI app and load model/tokenizer
app = FastAPI(title="IMDB Sentiment API")
MODEL_NAME = "UfukCem/imdb-bert-LoRA-finetuned"
logger.info("Model loading: %s", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
logger.info("Model loaded.")

class TextRequest(BaseModel):
    """Request model for text input."""
    text: str

@app.get("/")
def root():
    """Root endpoint to check if the API is running."""
    logger.info("Root endpoint runned.")
    return {"message": "IMDB Sentiment API is running"}


@app.post("/predict")
def predict(request: TextRequest):
    """Predict the sentiment of the given text."""
    start_time = datetime.now()
    logger.info("Prediction request received: %s", request.text)

    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

        labels = ["negative", "positive"]
        confidence = float(probs[0][pred])
        logger.info(
            "Prediction: %s | Trust: %.4f | Time: %.2fs",
            labels[pred],
            confidence,
            (datetime.now() - start_time).total_seconds(),
        )

        return {
            "label": labels[pred],
            "confidence": float(probs[0][pred]),
        }

    except Exception as e:
        logger.error("Eror Occurred: %s", e, exc_info=True)
        return {"error": str(e)}
