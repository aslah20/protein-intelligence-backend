from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

router = APIRouter()

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "app/models/binding/binding_model_ligand.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

# -----------------------------
# LOAD TOKENIZER + BASE MODEL
# -----------------------------
print("⏳ Loading ProtBERT (this may take time first run)...")

tokenizer = BertTokenizer.from_pretrained(
    "app/models/binding/full_model/",
    local_files_only=True
)

bert_model = BertModel.from_pretrained(
    "app/models/binding/full_model/",
    local_files_only=True
)

print("✅ ProtBERT loaded")

# -----------------------------
# DEFINE MODEL
# -----------------------------
class BindingModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        return self.classifier(x).squeeze(-1)

# -----------------------------
# LOAD YOUR WEIGHTS
# -----------------------------
model = BindingModel(bert_model).to(DEVICE)
#print("Skipping weights loading")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#print("Classifier weight mean:", model.classifier.weight.mean().item())
#print("Classifier weight std:", model.classifier.weight.std().item())
model.eval()

print("✅ Binding model ready")

# -----------------------------
# REQUEST MODEL
# -----------------------------
class BindingRequest(BaseModel):
    sequences: List[str]

# -----------------------------
# PREDICTION
# -----------------------------
def predict_binding(sequence: str):
    seq = sequence.replace("\n", "").replace(" ", "").strip()

    # IMPORTANT preprocessing
    seq_spaced = " ".join(list(seq))

    inputs = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        probs = torch.sigmoid(logits)

    # 🔍 DEBUG PRINTS
    print("\n=== DEBUG ===")
    print("Sequence length:", len(seq))
    print("PROBS SAMPLE:", probs[0][:20])
    print("MAX PROB:", probs.max().item())
    print("==============\n")

    probs = probs.squeeze()

    probs = probs[:len(seq)]

    #preds = (probs > 0.5).int().cpu().numpy()
    threshold = probs.mean().item()
    preds = (probs > threshold).int().cpu().numpy()

    binding_positions = [i + 1 for i, v in enumerate(preds) if v == 1]

    binding_count = len(binding_positions)
    length = len(seq)

    return {
        "sequence": seq,
        "binding": binding_count > 0,
        "binding_sites_count": binding_count,
        "binding_positions": binding_positions[:20],
        "binding_ratio": binding_count / length if length > 0 else 0
    }

# -----------------------------
# ENDPOINT
# -----------------------------
@router.post("/predict/binding/")
def predict_binding_batch(request: BindingRequest):
    return {
        "results": [predict_binding(seq) for seq in request.sequences]
    }