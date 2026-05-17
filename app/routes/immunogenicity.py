from fastapi import APIRouter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from pydantic import BaseModel

router = APIRouter()

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "app/models/immunogenicity/protbert_immuno2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL (once)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("✅ Immunogenicity model loaded")

# -----------------------------
# REQUEST MODEL
# -----------------------------
class BatchRequest(BaseModel):
    sequences: List[str]

# -----------------------------
# ENDPOINT
# -----------------------------
@router.post("/predict/immunogenicity/")
def predict_immunogenicity(request: BatchRequest):

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

        results.append({
            "sequence": seq,
            "prediction": int(predicted_class.item()),
            "confidence": float(confidence.item())
        })

    return {"results": results}