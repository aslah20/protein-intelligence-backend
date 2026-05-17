from fastapi import APIRouter, HTTPException
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
from typing import List
from pydantic import BaseModel

router = APIRouter()

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "asmaslah/proteinai-localization"
MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model (runs once at startup)
# -----------------------------
def load_localization_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model = model.to(device)
        model.eval()

        label_encoder = joblib.load(
            os.path.join(MODEL_PATH, "label_encoder.pkl")
        )

        print("✅ Localization model loaded successfully.")
        return tokenizer, model, label_encoder

    except Exception as e:
        print("❌ Error loading localization model:", e)
        raise e


tokenizer, model, label_encoder = load_localization_model()


# ----- REQUEST MODEL -----
class BatchLocalizationRequest(BaseModel):
    sequences: List[str]


# ----- BATCH ENDPOINT -----
@router.post("/predict/localization/")
def predict_localization(request: BatchLocalizationRequest):

    results = []

    for seq in request.sequences:
        clean_seq = " ".join(list(seq.strip()))

        inputs = tokenizer(
            clean_seq,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

            confidence, predicted_class = torch.max(probs, dim=1)

        predicted_label = label_encoder.inverse_transform(
            [predicted_class.item()]
        )[0]

        results.append({
            "sequence": seq,
            "prediction": predicted_label,
            "confidence": float(confidence.item())
        })

    return {"results": results}