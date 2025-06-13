from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

# load dataset
dataset = load_dataset("civil_comments", split="train[:5000]")

# preprocess
def preprocess(batch):
    return {
        "text": batch["text"],
        "label": [int(t > 0.5) for t in batch["toxicity"]]
    }

dataset = dataset.map(
    preprocess,
    batched=True,
    load_from_cache_file=True,
    desc="Processing dataset"
)
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["text", "label"]])
split = dataset.train_test_split(test_size=0.2, seed = 32)
train_dataset = split["train"]
test_dataset = split["test"]

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize in batches with truncation and padding
def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    load_from_cache_file=True,
    desc="Tokenizing train data"
)
test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    load_from_cache_file=True,
    desc="Tokenizing test data"
)

# Set format for PyTorch
torch_columns = ["input_ids", "attention_mask", "label"]
train_dataset.set_format(type="torch", columns=torch_columns)
test_dataset.set_format(type="torch", columns=torch_columns)

# load base model and fine tune with LoRA
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_lin", "v_lin"],  # common in BERT-like models
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    inference_mode=False
)
model = get_peft_model(base_model, peft_config)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    report_to="none",
    logging_steps=100,
    no_cuda=True
)

# Metric for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]  # pass as `labels`, not `label`
        )
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)

accuracy = correct / total
print(f"\nCustom Evaluation Accuracy: {accuracy:.4f}")


# model.save_pretrained("./outputs")
# tokenizer.save_pretrained("./outputs")
