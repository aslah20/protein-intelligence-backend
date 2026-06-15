from fastapi import APIRouter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from pydantic import BaseModel

router = APIRouter()

# -----------------------------
# CONFIG
# -----------------------------
#BASE_MODEL = "facebook/esm2_t6_8M_UR50D"
MODEL_REPO = "asmaslah/proteinai-stability"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.to(device)
model.eval()

print("✅ Stability model loaded")

# -----------------------------
# REQUEST MODEL
# -----------------------------
class BatchRequest(BaseModel):
    sequences: List[str]

# -----------------------------
# ENDPOINT
# -----------------------------
@router.post("/predict/stability/")
def predict_stability(request: BatchRequest):

    results = []

    for seq in request.sequences:

        clean_seq = seq.strip()

        inputs = tokenizer(
            clean_seq,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.item()

        results.append({
            "sequence": clean_seq,
            "prediction": float(prediction)
        })

    return {"results": results}