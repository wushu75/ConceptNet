# ConceptNet — Voice IP Infrastructure for AI

![Accuracy](https://img.shields.io/badge/classifier_accuracy-83%25-brightgreen)
![Languages](https://img.shields.io/badge/languages-9_live-blue)
![Dataset](https://img.shields.io/badge/dataset-730_examples-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![Patents](https://img.shields.io/badge/patents-pending-orange)
![Stage](https://img.shields.io/badge/stage-pre--seed-blue)
![Model](https://img.shields.io/badge/model-HuggingFace-yellow)

> The intent classification layer for enterprise AI agents. Token-free. 9 languages. No LLM required.

---

## Try it

| | |
|--|--|
| **Live Sandbox** | [conceptnet.co.uk/sandbox](https://conceptnet.co.uk/sandbox/) |
| **Benchmarks** | [conceptnet.co.uk/benchmarks](https://conceptnet.co.uk/benchmarks/) |
| **Hugging Face** | [huggingface.co/spaces/conceptnetUk/voice-ip-sandbox](https://huggingface.co/spaces/conceptnetUk/voice-ip-sandbox) |
| **Investor One-Pager** | [conceptnet.co.uk/docs/investor.html](https://conceptnet.co.uk/docs/investor.html) |
| **Website** | [conceptnet.co.uk](https://conceptnet.co.uk) |

---

## What it does

ConceptNet classifies enterprise voice and text commands into a structured 4-layer intent taxonomy and produces deterministic JSON output for agent execution — with no LLM call, no tokens, and no API dependency.

A 50-person enterprise team doing 20 queries per day pays **$44K/year** with ConceptNet vs **$125K/year** with GPT-4o. No hallucination. Sub-100ms latency. Runs locally.

---

## The 4-Layer Intent Taxonomy

The core invention is a novel hierarchical classification of enterprise voice commands into four layers, each with distinct agent execution semantics.

```
Layer 1 — Basic           "Do X"
Layer 2 — Context-Aware   "Do X when Y"
Layer 3 — Predictive      "Do X before Y"
Layer 4 — Autonomous      "Do X always / automatically"
```

**Layer 1 — Basic**
Single immediate action. No trigger, no condition, no scheduling.
> "Schedule a board meeting for Tuesday"
> "Send the Q3 report to the finance team"

**Layer 2 — Context-Aware**
Action triggered by a condition. Agent registers a listener and executes when the condition is met.
> "Send the report when the auditor signs off"
> "Create a Jira ticket if the error rate exceeds 5 percent"

**Layer 3 — Predictive**
Proactive action scheduled in anticipation of a future event.
> "Alert the account manager before the contract expires"
> "Prepare the board pack before the quarterly review"

**Layer 4 — Autonomous**
Persistent background agent. Deploys once, executes on every occurrence with no further prompting.
> "Automatically update Salesforce after every sales call"
> "Always send a follow-up email whenever a meeting ends"

---

## Architecture

```
Input text or voice (9 languages)
         │
         ▼
┌─────────────────────────────────────────┐
│  Fast Path — TF-IDF + Logistic         │
│  Regression                             │
│  83% coverage · <5ms · CPU only        │
└──────────────┬──────────────────────────┘
               │ confidence < threshold
               ▼
┌─────────────────────────────────────────┐
│  Neural Path — DistilBERT fine-tuned   │
│  (distilbert-base-multilingual-cased)  │
│  95%+ coverage · <100ms · GPU optional │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Constrained Grammar Output            │
│  Exactly 4 valid outputs               │
│  No hallucination possible             │
└──────────────┬──────────────────────────┘
               │
               ▼
    Structured JSON → Agent Execution
```

The two-stage architecture follows the statistical distillation pattern: an LLM labels training examples, those labels train a fast deterministic classifier, and the classifier handles high-frequency patterns at zero cost. The neural path catches novel phrasing the fast path misses. The constrained grammar ensures the model can only output one of four valid labels — deterministic, auditable, enterprise-safe.

---

## Languages

Live today: 🇬🇧 English · 🇫🇷 French · 🇪🇸 Spanish · 🇩🇪 German · 🇮🇹 Italian · 🇧🇷 Portuguese · 🇨🇳 Chinese · 🇸🇦 Arabic · 🇷🇺 Russian

The 4-layer taxonomy applies universally across all languages. Trigger keyword detection is implemented per language. The constrained grammar output is language-agnostic structured JSON. The multilingual neural model (`distilbert-base-multilingual-cased`) supports 104 languages out of the box.

---

## Repository Structure

```
ConceptNet/
├── README.md
├── LICENSE
├── sandbox/
│   └── index.html              # Live Voice IP Sandbox (obfuscated)
├── core/
│   ├── train_classifier.py     # Fast-path training pipeline
│   ├── train_neural.py         # Neural model fine-tuning script
│   └── intent_grammar.txt      # Constrained output schema
├── data/
│   ├── conceptnet_dataset_v2.json   # 730 labelled examples, 9 languages
│   └── conceptnet_dataset_v2.csv    # Same dataset in CSV format
└── docs/
    └── ARCHITECTURE.md         # Full technical architecture
```

---

## Dataset

`data/conceptnet_dataset_v2.json` — 730 labelled enterprise voice command examples across 9 languages and all 4 intent layers.

| Language | L1 Basic | L2 Context | L3 Predictive | L4 Autonomous | Total |
|----------|----------|------------|---------------|---------------|-------|
| English  | 102 | 100 | 99 | 99 | **400** |
| French   | 15 | 10 | 10 | 10 | **45** |
| Spanish  | 15 | 10 | 10 | 10 | **45** |
| German   | 10 | 10 | 10 | 10 | **40** |
| Italian  | 10 | 10 | 10 | 10 | **40** |
| Portuguese | 10 | 10 | 10 | 10 | **40** |
| Chinese  | 10 | 10 | 10 | 10 | **40** |
| Arabic   | 10 | 10 | 10 | 10 | **40** |
| Russian  | 10 | 10 | 10 | 10 | **40** |
| **Total** | **192** | **180** | **179** | **179** | **730** |

Each example includes: `text`, `label`, `intent_layer`, `language`, `tool`, `action`, `condition`, `prediction`, `autonomous`.

---

## Fast-Path Classifier Results

Trained on 80% of the dataset, tested on 20%.

```
              precision    recall    f1-score
Basic             0.66      0.81       0.73
Context-Aware     0.90      0.78       0.84
Predictive        1.00      0.88       0.94
Autonomous        0.84      0.93       0.88

accuracy                               0.83
5-fold CV: 0.847 ± 0.018
```

Layer 3 Predictive achieves 100% precision — the "before" trigger pattern is unambiguous across all 9 languages.

---

## Quick Start

```bash
git clone https://github.com/wushu75/ConceptNet
cd ConceptNet
pip install scikit-learn
python core/train_classifier.py
```

To fine-tune the neural model (requires GPU or Google Colab):

```bash
pip install transformers torch datasets accelerate evaluate
python core/train_neural.py
```

---

## Structured Output Format

Every classification produces structured JSON ready for agent execution:

```json
{
  "text": "Send the report when the auditor signs off",
  "intent_layer": 2,
  "intent_label": "Context-Aware",
  "execution_mode": "conditional",
  "tool": "EMAIL",
  "action": "send_message",
  "condition": "auditor signs off",
  "prediction": null,
  "autonomous": false,
  "confidence": 0.94
}
```

```json
{
  "text": "Automatically update Salesforce after every sales call",
  "intent_layer": 4,
  "intent_label": "Autonomous",
  "execution_mode": "autonomous",
  "tool": "CRM",
  "action": "update_record",
  "condition": null,
  "prediction": null,
  "autonomous": true,
  "confidence": 0.97
}
```

---

## Voice IP Stacking

The classification layer powers the Voice IP Stacking platform — a novel model by which enterprises build, own, and license their own voice intent IP.

```
① Privatise  →  Enterprise uploads voice workflow data. They own it.
② Stack      →  Data classified across 4 intent layers. IP compounds.
③ Own        →  Labelled dataset = proprietary IP. Switching cost = losing it.
④ License    →  Enterprise licenses IP to others. ConceptNet takes 10%.
```

This creates a platform flywheel: more users → more data → more IP transactions → more revenue → better model → more users.

---

## Benchmarks

| Model | Cost per query | Latency | Languages | Token-free |
|-------|---------------|---------|-----------|------------|
| GPT-4o | $0.08 | 800ms | ~50 | ✗ |
| Claude Sonnet | $0.06 | 600ms | ~30 | ✗ |
| **ConceptNet** | **$0.015** | **<100ms** | **200+** | **✓** |

Full benchmarks: [conceptnet.co.uk/benchmarks](https://conceptnet.co.uk/benchmarks/)

---

## Raising

ConceptNet is raising **£1M pre-seed** at **£5–6M pre-money**.

- EIS eligible — 30% tax relief for UK investors
- SEIS eligible — 50% tax relief on first £250K
- Based in London, UK
- 4-person technical team
- Patents pending on the 4-layer taxonomy and Voice IP Stacking architecture

**Contact:** tonymomoh@icloud.com · 07733 246865
**Investor one-pager:** [conceptnet.co.uk/docs/investor.html](https://conceptnet.co.uk/docs/investor.html)

---

## Licence

The code in this repository is released under the **MIT Licence**.

The ConceptNet 4-layer intent taxonomy, Voice IP Stacking architecture, and associated datasets are proprietary intellectual property of ConceptNet Ltd. Patents pending.

© 2026 ConceptNet Ltd · London, UK · All rights reserved
