# ConceptNet — Voice Intent Infrastructure for Enterprise AI

> The intent classification layer for enterprise AI agents, government workflow automation, and autonomous defence systems.

**Raising £2M Seed · EIS Eligible · Kings Cross, London · [conceptnet.co.uk](https://conceptnet.co.uk)**

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

**Token-free. 9 languages. No LLM required. Locally deployable. Air-gap capable.**

### The 4-Layer Intent Taxonomy

Every enterprise voice command — in any language, any industry, any country — fits into exactly one of these:

| Layer | Pattern | Execution | Example |
|-------|---------|-----------|---------|
| **L1 Basic** | "Do X" | Immediate | "Schedule a board meeting" |
| **L2 Context-Aware** | "Do X when Y" | Conditional trigger | "Send report when contract is signed" |
| **L3 Predictive** | "Do X before Y" | Proactive | "Alert manager before deadline expires" |
| **L4 Autonomous** | "Do X always" | Persistent agent | "Auto-update CRM after every call" |

Nobody had defined this taxonomy. Nobody had built a dedicated classifier for it. We did. It's patented.

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

### Why This Is Defensible

| Property | What it means |
|---------|--------------|
| Constrained grammar | Exactly 4 outputs — hallucination mathematically impossible |
| Token-free | No US cloud dependency — 3× cheaper than GPT-4o |
| Air-gap capable | No internet required — defence and government grade |
| Locally deployable | Data never leaves the organisation |
| Patents pending | Novel taxonomy — no prior art |

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
| Intent layers | **All 4** — balanced distribution |
| L3 surface forms | "before", "ahead of", "in advance of", "prior to", "in time for", "by the time" |
| Format | JSON + CSV |

---

## Plugin Ecosystem

ConceptNet is available as a plugin for all major Chinese AI agent frameworks:

| Platform | Plugin | Community |
|---------|--------|-----------|
| DeepSeek Harness | [conceptnet-dsh-plugin](https://github.com/wushu75/conceptnet-dsh-plugin) | [Discussion #5559](https://github.com/deepseek-ai/deepseek-harness/discussions/5559) — 211K ⭐ |
| Qwen / Alibaba | [conceptnet-qwen-plugin](https://github.com/wushu75/conceptnet-qwen-plugin) | QwenLM ecosystem |
| Kimi / Moonshot | [conceptnet-kimi-plugin](https://github.com/wushu75/conceptnet-kimi-plugin) | Moonshot AI ecosystem |
| GLM / Zhipu | [conceptnet-glm-plugin](https://github.com/wushu75/conceptnet-glm-plugin) | THUDM ecosystem |
| Doubao / ByteDance | [conceptnet-doubao-plugin](https://github.com/wushu75/conceptnet-doubao-plugin) | ByteDance ecosystem |

---

## Try It

| Asset | Link |
|-------|------|
| **Live Sandbox** | [conceptnet.co.uk/sandbox/](https://conceptnet.co.uk/sandbox/) |
| **Hugging Face Model** | [huggingface.co/conceptnetUk/intent-classifier](https://huggingface.co/conceptnetUk/intent-classifier) |
| **HF Space** | [huggingface.co/spaces/conceptnetUk/voice-ip-sandbox](https://huggingface.co/spaces/conceptnetUk/voice-ip-sandbox) |
| **API Documentation** | [docs/API.md](docs/API.md) |
| **Investor One-Pager** | [conceptnet.co.uk/docs/investor.html](https://conceptnet.co.uk/docs/investor.html) |

---

## Traction

| Metric | Number |
|--------|--------|
| GitHub clones | **188** from **75 unique developers** |
| HF model downloads | **17+** |
| Independent peer review | ✅ Adversarial holdout confirmed |
| API token requests | Active — first within minutes of announcing |
| Plugin ecosystem | 5 Chinese AI platforms |
| Government pipeline | Nigeria FIRS, Mauritius, Rivers State, Qatar, Ivory Coast |
| Defence | UK Defence Innovation submitted — DIOL233749 |
| Built before raise | **£448,000 / $600,000** |

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
│   ├── API.md
│   └── ARCHITECTURE.md
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
- **Classifier logic** — production inference code not in this repo
- **Dataset** — released for research only, commercial use requires licence
- **© 2026 ConceptNet Ltd** — all rights reserved

---

## Voice IP Stacking — The Platform Model

The model that makes ConceptNet a 10-year moat:

```
① PRIVATISE  → Enterprise data encrypted. They own it entirely.
② STACK      → Classified across 4 intent layers. Compounds over time.
③ OWN        → Classified dataset = proprietary IP. Switching = losing it.
④ LICENSE    → Enterprises license IP to others. ConceptNet takes 10%.
```

---

## Raising

| Item | Detail |
|------|--------|
| Amount | **£2M Seed** |
| Pre-money | **£15M** |
| EIS eligible | **30% tax relief** for UK investors |
| Built before raise | **$600,000 / £448,000** |
| CTO | Tim Storey — confirmed fractional |
| Defence | DASA DIOL233749 submitted |
| Contact | tonymomoh@icloud.com · 07733 246865 |

---

## Languages Live

🇬🇧 English · 🇫🇷 French · 🇪🇸 Spanish · 🇩🇪 German · 🇮🇹 Italian · 🇧🇷 Portuguese · 🇨🇳 Chinese · 🇸🇦 Arabic · 🇷🇺 Russian

---

*ConceptNet Ltd · Kings Cross, London · © 2026 · Patents pending*
*tonymomoh@icloud.com · 07733 246865 · conceptnet.co.uk*
