from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

router = APIRouter()

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app/models/ptm/PTM_Model_Acce_last")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

# -----------------------------
# LOAD MODEL
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("✅ PTM model loaded")

# -----------------------------
# REQUEST
# -----------------------------
class PTMRequest(BaseModel):
    sequences: List[str]

# -----------------------------
# PREDICTION
# -----------------------------
def predict_ptm(sequence: str):
    seq = sequence.strip()

    window_size = 15
    ptm_positions = []
    ptm_probs = []

    for i, aa in enumerate(seq):
        if aa != "K":
            continue

        # LEFT
        left = seq[max(0, i - window_size):i]
        left = "X" * (window_size - len(left)) + left

        # RIGHT
        right = seq[i+1:i+1+window_size]
        right = right + "X" * (window_size - len(right))

        window = left + "K" + right  # 31 length

        inputs = tokenizer(
            window,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=35
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        prob = probs[0][1].item()  # probability of acetylation

        if prob > 0.5:
            ptm_positions.append(i + 1)
            ptm_probs.append(prob)

    return {
        "sequence": seq,
        "ptm": len(ptm_positions) > 0,
        "ptm_sites_count": len(ptm_positions),
        "ptm_positions": ptm_positions[:20],
        "ptm_ratio": len(ptm_positions) / len(seq) if len(seq) > 0 else 0
    }

# -----------------------------
# ENDPOINT
# -----------------------------
@router.post("/predict/ptm/")
def predict_ptm_batch(request: PTMRequest):
    return {
        "results": [predict_ptm(seq) for seq in request.sequences]
    }
