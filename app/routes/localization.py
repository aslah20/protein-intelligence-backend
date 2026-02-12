from fastapi import APIRouter, HTTPException
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os

router = APIRouter()

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "asmaslah/protein-localization-protbert"
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


# -----------------------------
# Prediction Endpoint
# -----------------------------
@router.post("/")
def predict_localization(sequence: str):
    try:
        if not sequence or len(sequence.strip()) == 0:
            raise HTTPException(status_code=400, detail="Sequence cannot be empty.")

        # ProtBERT expects space-separated amino acids
        formatted_seq = " ".join(list(sequence.strip().upper()))

        inputs = tokenizer(
            formatted_seq,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()

        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(probabilities[0][predicted_index])

        return {
            "task": "Protein Localization",
            "prediction": predicted_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
