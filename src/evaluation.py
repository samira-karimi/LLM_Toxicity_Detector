import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and base model name
model_path = "./outputs"
peft_config = PeftConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, model_path)
model.to(device)
model.eval()

# Load Civil Comments dataset test split
dataset = load_dataset("civil_comments", split="test")

# Remove rows with missing 'toxicity'
dataset = dataset.filter(lambda x: x["toxicity"] is not None)

# Preprocessing
MAX_LENGTH = 128

def preprocess(batch):
    tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    tokenized["label"] = [1 if float(t) > 0.5 else 0 for t in batch["toxicity"]]
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_loader = DataLoader(dataset, batch_size=16)

# Evaluation loop
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)

accuracy = correct / total
print(f"\nEvaluation Accuracy: {accuracy:.4f}")
