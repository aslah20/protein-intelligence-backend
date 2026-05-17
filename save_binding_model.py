from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

MODEL_PATH = "app/models/binding/binding_model_ligand.pt"

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
bert_model = BertModel.from_pretrained("Rostlab/prot_bert")

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

model = BindingModel(bert_model)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# SAVE FULL MODEL
model.bert.save_pretrained("app/models/binding/full_model/")
tokenizer.save_pretrained("app/models/binding/full_model/")

print("✅ Model saved locally!")