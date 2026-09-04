# ConceptNet — Voice IP Infrastructure for AI

> The intent classification layer for enterprise AI agents and government workflow automation.

**Raising £1M Pre-Seed · EIS Eligible · London, UK · [conceptnet.co.uk](https://conceptnet.co.uk)**

[![Live Sandbox](https://img.shields.io/badge/sandbox-live-10B981?style=flat-square)](https://conceptnet.co.uk/sandbox/)
[![HuggingFace](https://img.shields.io/badge/model-HuggingFace-yellow?style=flat-square)](https://huggingface.co/conceptnetUk/intent-classifier)
[![Accuracy](https://img.shields.io/badge/accuracy-100%25-10B981?style=flat-square)](https://huggingface.co/conceptnetUk/intent-classifier)
[![Languages](https://img.shields.io/badge/languages-9-2563EB?style=flat-square)](https://conceptnet.co.uk/sandbox/)
[![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)
[![Patents](https://img.shields.io/badge/patents-pending-F59E0B?style=flat-square)](#ip-protection)

---

## Independent Peer Review — August 2026

An independent ML researcher from the Hugging Face community ran adversarial holdout tests designed to catch fake accuracy numbers. He tried to break it.

| Test | Result |
|------|--------|
| Standard test set | **100%** (epochs 4 and 5) |
| Adversarial holdout | **99.3%** confirmed |
| Grouped lexical-family holdout | **99.78%** |
| Issues found | 4 |
| Issues fixed | **All 4 — within 24 hours** |

> *"The obvious train/test leakage explanation did not survive that check."*
> — Independent researcher, Hugging Face community

[Read the full evaluation →](https://discuss.huggingface.co/t/conceptnet-4-layer-enterprise-voice-intent-classifier-98-6-accuracy-9-languages-token-free-open-source/179274)

---

## What is ConceptNet?

ConceptNet classifies enterprise voice and text commands into 4 intent layers — producing structured JSON for agent execution automatically.

**Token-free. 9 languages. No LLM required. Locally deployable.**

### The 4-Layer Intent Taxonomy

Every enterprise voice command — in any language, any industry, any country — fits into exactly one of these:

| Layer | Pattern | Execution | Example |
|-------|---------|-----------|---------|
| **L1 Basic** | "Do X" | Immediate | "Schedule a board meeting" |
| **L2 Context-Aware** | "Do X when Y" | Conditional trigger | "Send report when contract is signed" |
| **L3 Predictive** | "Do X before Y" | Proactive | "Alert manager before deadline expires" |
| **L4 Autonomous** | "Do X always" | Persistent agent | "Auto-update CRM after every call" |

---

## Architecture — Two-Stage Pipeline

```
Input (voice or text — 9 languages)
         ↓
FAST PATH — TF-IDF + Logistic Regression
83% coverage · <5ms · CPU only · zero cost
         ↓ if confidence below threshold
NEURAL PATH — DistilBERT fine-tuned
100% accuracy · <100ms · GPU optional
         ↓ constrained by
GRAMMAR LAYER — exactly 4 valid outputs
No hallucination · Deterministic · Auditable
         ↓
Structured JSON → Agent execution → Enterprise tools
```

### Cascade Performance

| Threshold | Fast coverage | Fast accuracy | Final accuracy |
|-----------|--------------|---------------|----------------|
| 0.50 | 69.2% | 95.0% | 95.9% |
| 0.55 | 60.3% | 98.9% | 99.3% |
| 0.65 | 43.2% | 100% | 100% |

### Layer Precedence
Mixed semantics: **L4 > L3 > L2 > L1**

---

## Structured Output

```json
{
  "text": "Send the report when the contract is signed",
  "intent_layer": 2,
  "intent_label": "Context-Aware",
  "execution_mode": "conditional",
  "tool": "EMAIL",
  "action": "send_message",
  "condition": "contract is signed",
  "confidence": 0.94,
  "language": "en",
  "latency_ms": 3
}
```

---

## Dataset

| Metric | Value |
|--------|-------|
| Total examples | **757** |
| Languages | **9** — EN, FR, ES, DE, IT, PT, ZH, AR, RU |
| Intent layers | **All 4** |
| L3 surface forms | "before", "ahead of", "in advance of", "prior to", "in time for", "by the time" |
| Format | JSON + CSV |

---

## Try It

| Asset | Link |
|-------|------|
| **Live Sandbox** | [conceptnet.co.uk/sandbox/](https://conceptnet.co.uk/sandbox/) |
| **Hugging Face Model** | [huggingface.co/conceptnetUk/intent-classifier](https://huggingface.co/conceptnetUk/intent-classifier) |
| **HF Space** | [huggingface.co/spaces/conceptnetUk/voice-ip-sandbox](https://huggingface.co/spaces/conceptnetUk/voice-ip-sandbox) |
| **Benchmarks** | [conceptnet.co.uk/benchmarks/](https://conceptnet.co.uk/benchmarks/) |
| **Website** | [conceptnet.co.uk](https://conceptnet.co.uk) |

---

## Repo Structure

```
ConceptNet/
├── README.md
├── LICENSE                    # MIT
├── CNAME                      # conceptnet.co.uk
├── requirements.txt
├── index.html                 # Website homepage
├── privacy.html
├── terms.html
├── core/
│   ├── train_classifier.py    # Fast-path training
│   ├── train_neural.py        # Neural model training
│   └── intent_grammar.txt     # Constrained grammar
├── data/
│   ├── conceptnet_dataset_v2.json
│   └── conceptnet_dataset_v2.csv
├── docs/
│   ├── investor.html
│   ├── ARCHITECTURE.md
│   └── star.html
├── sandbox/
│   └── index.html             # Live sandbox
├── benchmarks/
├── examples/
│   └── quickstart.py
└── api/
```

---

## IP Protection

- **Patents pending** — 4-layer taxonomy, Voice IP Stacking, constrained grammar architecture
- **Classifier logic obfuscated** — production inference code not in this repo
- **Dataset** — released for research only, commercial use requires licence
- **© 2026 ConceptNet Ltd** — all rights reserved

---

## Voice IP Stacking

The platform model that makes ConceptNet a 10-year moat:

```
① PRIVATISE  → Enterprise data encrypted. They own it entirely.
② STACK      → Classified across 4 intent layers. Compounds over time.
③ OWN        → Classified dataset = proprietary IP. Switching = losing it.
④ LICENSE    → Enterprises license IP to others. ConceptNet takes 10%.
```

---

## Plugin Ecosystem

| Platform | Plugin | Stars |
|---------|--------|-------|
| DeepSeek Harness | [conceptnet-dsh-plugin](https://github.com/wushu75/conceptnet-dsh-plugin) | 211K |
| Qwen / Alibaba | [conceptnet-qwen-plugin](https://github.com/wushu75/conceptnet-qwen-plugin) | — |
| Kimi / Moonshot | [conceptnet-kimi-plugin](https://github.com/wushu75/conceptnet-kimi-plugin) | — |
| GLM / Zhipu | [conceptnet-glm-plugin](https://github.com/wushu75/conceptnet-glm-plugin) | — |
| Doubao / ByteDance | [conceptnet-doubao-plugin](https://github.com/wushu75/conceptnet-doubao-plugin) | — |

---

## Traction

- 🤗 **17 downloads** on Hugging Face
- ⭐ **99 clones** from **53 unique developers** in 14 days
- 🔬 **Independently verified** by ML community — adversarial holdout confirmed
- 🌍 **Pilot conversations** across UK, Africa, Middle East
- 📰 **Published** on Medium and Substack — 5,500+ followers

---

## Raising

- **Amount:** £2M Pre-Seed
- **Pre-money:** £5–8M
- **EIS eligible:** 30% tax relief for UK investors
- **Built before raise:** $600,000 / £408,000
- **Contact:** tonymomoh@icloud.com · 07733 246865

---

## Languages Live

🇬🇧 English · 🇫🇷 French · 🇪🇸 Spanish · 🇩🇪 German · 🇮🇹 Italian · 🇧🇷 Portuguese · 🇨🇳 Chinese · 🇸🇦 Arabic · 🇷🇺 Russian

---

*ConceptNet Ltd · Kings Cross, London · © 2026 · Patents pending*
*tonymomoh@icloud.com · 07733 246865 · conceptnet.co.uk*
