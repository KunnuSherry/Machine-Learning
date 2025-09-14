# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Load artifacts
model = joblib.load("spam_clf.joblib")
# Note: The model is a pipeline that includes vectorization, so we don't need the separate vectorizer

app = FastAPI(title="SMS Spam Classifier")

# Allow React dev server
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Message) -> Dict:
    text = payload.message
    # The model is a pipeline, so we pass the raw text directly
    pred = int(model.predict([text])[0])
    probs = model.predict_proba([text])[0]  # [prob_ham, prob_spam] if labels are 0/1
    probability = float(probs[pred])

    label = "spam" if pred == 1 else "ham"
    return {
        "prediction": label,
        "prediction_label": pred,
        "probability": probability,
        "probabilities": probs.tolist()
    }
