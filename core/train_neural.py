
# Run this separately if you have a GPU or Apple Silicon:
# pip install transformers torch datasets accelerate

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np, json, evaluate

with open("conceptnet_dataset_v2.json") as f:
    data = json.load(f)

texts  = [d["text"] for d in data]
labels = [d["intent_layer"] - 1 for d in data]

LABEL_NAMES = ["Basic", "Context-Aware", "Predictive", "Autonomous"]
MODEL_NAME  = "distilbert-base-multilingual-cased"  # 104 languages, 134M params

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label={i: l for i, l in enumerate(LABEL_NAMES)},
    label2id={l: i for i, l in enumerate(LABEL_NAMES)},
)

dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2, seed=42)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

args = TrainingArguments(
    output_dir="./conceptnet-intent-classifier",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./conceptnet-intent-classifier")
tokenizer.save_pretrained("./conceptnet-intent-classifier")
print("Neural model saved. Upload to HF: huggingface-cli upload conceptnetUk/intent-classifier ./conceptnet-intent-classifier")
