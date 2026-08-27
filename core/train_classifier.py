import os
"""
ConceptNet Intent Classifier — Training Pipeline
Trains a small fine-tuned model on the 4-layer intent taxonomy.

Architecture:
  - Base model: distilbert-base-multilingual-cased (134M params, 104 languages)
  - Task: 4-class sequence classification (Basic/Context/Predictive/Autonomous)
  - Output: constrained to exactly 4 labels — fully deterministic
  - Target: sub-100ms inference, runs locally, zero API dependency

Usage:
  pip install transformers datasets scikit-learn torch
  python train_classifier.py

After training:
  - Model saved to ./conceptnet-intent-classifier/
  - Upload to Hugging Face Hub: huggingface.co/conceptnetUk/intent-classifier
"""

import json
import numpy as np
from pathlib import Path

# ── Load dataset ──────────────────────────────────────────────────────────
print("Loading dataset...")
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "conceptnet_dataset_v2.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

texts  = [d["text"] for d in data]
labels = [d["intent_layer"] - 1 for d in data]  # 0-indexed: 0=Basic, 1=Context, 2=Predictive, 3=Autonomous

LABEL_NAMES = ["Basic", "Context-Aware", "Predictive", "Autonomous"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL    = {i: l for i, l in enumerate(LABEL_NAMES)}

print(f"  Total examples: {len(texts)}")
for i, name in enumerate(LABEL_NAMES):
    count = labels.count(i)
    print(f"  Layer {i+1} ({name}): {count}")

# ── Fast path: Logistic Regression baseline ───────────────────────────────
# This is the "statistical distillation" approach the investor described.
# Train on TF-IDF features — interpretable, fast, no GPU needed.
# Use this as the fast path; fall back to the transformer for unknowns.

print("\nTraining fast-path logistic regression classifier...")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report
    import pickle

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        sublinear_tf=True,
        analyzer="word",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    print("\nFast-path classifier results:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Cross-validation
    cv_scores = cross_val_score(clf, vectorizer.transform(texts), labels, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save fast-path model
    Path("./conceptnet-intent-classifier").mkdir(exist_ok=True)
    with open("./conceptnet-intent-classifier/fast_path_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("./conceptnet-intent-classifier/fast_path_clf.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("\nFast-path model saved to ./conceptnet-intent-classifier/")

    # Show most informative features per class
    print("\nTop keywords per layer (fast-path explanation):")
    feature_names = vectorizer.get_feature_names_out()
    for i, name in enumerate(LABEL_NAMES):
        top_idx = np.argsort(clf.coef_[i])[-10:][::-1]
        top_features = [feature_names[j] for j in top_idx]
        print(f"  L{i+1} {name}: {', '.join(top_features)}")

except ImportError:
    print("scikit-learn not installed. Run: pip install scikit-learn")

# ── Full model: DistilBERT fine-tuning ────────────────────────────────────
print("\n" + "="*60)
print("Neural model training (requires: pip install transformers torch datasets)")
print("="*60)

NEURAL_TRAINING_SCRIPT = '''
# Run this separately if you have a GPU or Apple Silicon:
# pip install transformers torch datasets accelerate

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np, json, evaluate

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "conceptnet_dataset_v2.json")) as f:
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
'''

with open("./conceptnet-intent-classifier/train_neural.py", "w") as f:
    f.write(NEURAL_TRAINING_SCRIPT)
print("Neural training script saved to ./conceptnet-intent-classifier/train_neural.py")

# ── Constrained grammar inference ─────────────────────────────────────────
print("\n" + "="*60)
print("Constrained grammar approach (what the investor described)")
print("="*60)

CONSTRAINED_GRAMMAR = """
# ConceptNet Intent Grammar — Constrained Output Schema
# Exactly 4 valid outputs — no hallucination possible

GRAMMAR := LAYER_1 | LAYER_2 | LAYER_3 | LAYER_4

LAYER_1 := {
    "intent_layer": 1,
    "intent_label": "Basic",
    "trigger": null,
    "execution_mode": "immediate"
}

LAYER_2 := {
    "intent_layer": 2,
    "intent_label": "Context-Aware",
    "trigger": CONDITION_CLAUSE,
    "execution_mode": "conditional"
}

LAYER_3 := {
    "intent_layer": 3,
    "intent_label": "Predictive",
    "trigger": PREDICTION_CLAUSE,
    "execution_mode": "anticipatory"
}

LAYER_4 := {
    "intent_layer": 4,
    "intent_label": "Autonomous",
    "trigger": REPEAT_CLAUSE,
    "execution_mode": "autonomous"
}

CONDITION_CLAUSE  := "when" TEXT | "if" TEXT | "after" TEXT | "once" TEXT
PREDICTION_CLAUSE := "before" TEXT | "in advance of" TEXT
REPEAT_CLAUSE     := "always" | "automatically" | "every time" | "whenever" | "continuously"
"""

with open("./conceptnet-intent-classifier/intent_grammar.txt", "w") as f:
    f.write(CONSTRAINED_GRAMMAR)
print("Intent grammar saved. This is the constrained output schema for the specialist model.")
print("\nNext step: Fine-tune phi-2 or mistral-7b with this grammar as the output constraint.")
print("Library to use: outlines (github.com/outlines-dev/outlines)")
