"""A FastAPI application for sentiment analysis using a fine-tuned BERT
model on the IMDB dataset."""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="IMDB Sentiment API")
MODEL_NAME = "UfukCem/imdb-bert-LoRA-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


class TextRequest(BaseModel):
    """Request model for text input."""

    text: str


@app.get("/")
def root():
    """Root endpoint to check if the API is running."""
    return {"message": "IMDB Sentiment API is running"}


@app.post("/predict")
def predict(request: TextRequest):
    """Predict the sentiment of the given text."""
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    labels = ["negative", "positive"]
    return {
        "label": labels[pred],
        "confidence": float(probs[0][pred]),
    }
