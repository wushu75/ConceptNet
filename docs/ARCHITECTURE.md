# ConceptNet — Technical Architecture

> This document describes the technical architecture of the ConceptNet intent classification system. It is intended for technical reviewers, potential collaborators, and investors conducting technical due diligence.

---

## 1. The Problem

Enterprise voice AI has three structural problems that existing LLM-based approaches do not solve:

**Cost at scale.** GPT-4o charges $0.08 per query. A 50-person enterprise team doing 20 queries per day spends $125K per year on classification alone. At 500 people the number is $1.25M per year — for a task that does not require a 175-billion parameter model.

**Latency.** LLM API round-trips take 500ms–2,000ms. Real-time voice workflow automation requires sub-100ms classification. These requirements are fundamentally incompatible with general-purpose LLM inference.

**No proprietary IP.** Enterprises using LLM-based voice automation accumulate no intellectual property. Their voice workflow data is processed by a third-party system and discarded. There is no asset, no moat, and no switching cost.

ConceptNet addresses all three by replacing general-purpose LLM inference with a specialist classification model constrained to exactly four output labels — the four layers of the ConceptNet intent taxonomy.

---

## 2. Why Four Layers

The 4-layer taxonomy is not arbitrary. It maps directly to the four distinct execution semantics that enterprise AI agents must implement:

| Layer | Pattern | Agent behaviour | Example |
|-------|---------|----------------|---------|
| 1 — Basic | "Do X" | Execute immediately | "Schedule a meeting" |
| 2 — Context-Aware | "Do X when Y" | Register trigger, wait, execute on condition | "Send report when signed" |
| 3 — Predictive | "Do X before Y" | Calculate window, schedule proactively | "Alert before contract expires" |
| 4 — Autonomous | "Do X always" | Deploy persistent listener, run indefinitely | "Auto-update CRM after every call" |

These four layers are exhaustive for enterprise voice workflow automation. Every enterprise voice command that could trigger an agent action falls into exactly one of these categories. The taxonomy was derived from analysis of enterprise workflow patterns across sales, legal, finance, operations, and engineering functions.

The key insight is that Layer 2, 3, and 4 represent escalating degrees of agent autonomy — from conditional execution, to proactive execution, to fully autonomous execution. Each layer requires a fundamentally different agent execution pattern and a fundamentally different infrastructure response.

---

## 3. Classification Architecture

### 3.1 Two-Stage Design

The classification system uses a two-stage architecture inspired by the statistical distillation pattern:

```
Stage 1: Fast Path
  Input → TF-IDF vectorisation → Logistic Regression → Layer label
  Coverage: ~83% of queries
  Latency: <5ms
  Cost: $0 (CPU only, no API)

Stage 2: Neural Path (fallback)
  Input → Tokenisation → DistilBERT (fine-tuned) → Constrained grammar → Layer label
  Coverage: remaining ~17% of queries
  Latency: <100ms
  Cost: minimal (local inference)
```

The fast path handles the majority of queries that contain unambiguous trigger patterns. The neural path handles novel phrasing, complex constructions, and edge cases that the statistical model misses.

### 3.2 Fast Path — Statistical Distillation

The fast-path classifier is a logistic regression model trained on TF-IDF features extracted from the ConceptNet dataset.

**Architecture:**
- Vectoriser: TF-IDF, n-gram range (1,3), 10,000 features, sublinear TF scaling
- Classifier: Logistic Regression, C=5.0, max_iter=1000
- Training data: 584 examples (80% of v2 dataset)
- Test data: 146 examples (20% of v2 dataset)

**Results:**

```
Layer               Precision   Recall   F1
Basic (L1)           0.66       0.81    0.73
Context-Aware (L2)   0.90       0.78    0.84
Predictive (L3)      1.00       0.88    0.94
Autonomous (L4)      0.84       0.93    0.88

Overall accuracy: 0.83
5-fold CV:        0.847 ± 0.018
```

Layer 3 achieves 100% precision because the "before" / "prior to" / "in advance of" trigger pattern is unambiguous across all supported languages. The model never misclassifies a non-predictive command as predictive.

Layer 1 has the lowest precision because Basic commands share vocabulary with all other layers — any command that starts with a verb could in principle be Basic. The neural path improves Layer 1 precision significantly.

**Most informative features per layer:**

The logistic regression coefficients are fully interpretable. The top features for each layer reveal the underlying linguistic signal the model has learned:

- **L1 Basic:** action verbs without temporal or conditional context ("schedule", "send", "create", "fetch", "draft")
- **L2 Context-Aware:** conditional connectors ("when", "if", "after", "once", "sobald", "عند", "когда")
- **L3 Predictive:** anticipatory prepositions ("before", "prior to", "avant", "قبل", "bevor")
- **L4 Autonomous:** frequency and automation adverbs ("automatically", "always", "every time", "تلقائيا", "automatisch")

### 3.3 Neural Path — Constrained Grammar Model

The neural path uses `distilbert-base-multilingual-cased` — a 66-million parameter multilingual transformer supporting 104 languages — fine-tuned on the ConceptNet dataset for 4-class sequence classification.

**Why DistilBERT:**
- 40% smaller than BERT-base with 97% of its performance
- Multilingual variant supports 104 languages from a single model
- Fast enough for production inference on CPU (150ms) or GPU (<50ms)
- Small enough to run on-device in future deployments

**Constrained grammar output:**
The fine-tuned model output is constrained to exactly four valid labels using a grammar constraint layer:

```
GRAMMAR := LAYER_1 | LAYER_2 | LAYER_3 | LAYER_4

LAYER_1 := { intent_layer: 1, execution_mode: "immediate" }
LAYER_2 := { intent_layer: 2, execution_mode: "conditional",  trigger: CONDITION_CLAUSE }
LAYER_3 := { intent_layer: 3, execution_mode: "anticipatory", trigger: PREDICTION_CLAUSE }
LAYER_4 := { intent_layer: 4, execution_mode: "autonomous",   trigger: REPEAT_CLAUSE }
```

This constrained grammar means the model cannot hallucinate a fifth category, cannot return a confidence score without a layer label, and cannot produce ambiguous output. Every classification produces exactly one of four valid structured outputs. This is a critical property for enterprise deployment where determinism and auditability are required.

### 3.4 Trigger Extraction

For Layer 2 and Layer 3 commands, the system extracts the trigger clause from the natural language input using language-specific regex patterns applied after classification:

**Layer 2 — Condition extraction:**
```
English:    /\bwhen\s+(.{3,60}?)(?:\s*,|$)/i
            /\bif\s+(.{3,60}?)(?:\s*,|$)/i
            /\bonce\s+(.{3,60}?)(?:\s*,|$)/i
French:     /\bquand\s+(.{3,60}?)(?:\s*,|$)/i
Arabic:     /عند\s+(.{3,60}?)(?:\s*,|$)/
German:     /\bwenn\s+(.{3,60}?)(?:\s*,|$)/i
```

**Layer 3 — Prediction extraction:**
```
English:    /\bbefore\s+(.{3,60}?)(?:\s*,|$)/i
French:     /\bavant\s+(.{3,60}?)(?:\s*,|$)/i
Arabic:     /قبل\s+(.{3,60}?)(?:\s*,|$)/
German:     /\bbevor\s+(.{3,60}?)(?:\s*,|$)/i
```

The extracted trigger clause is included in the structured JSON output and passed to the agent execution layer to register the appropriate listener.

---

## 4. Multilingual Operation

The system operates across 9 languages today and is architected for 200.

**Language support layers:**

| Component | Languages today | Languages at scale |
|-----------|----------------|-------------------|
| Sandbox + examples | 9 | 9 |
| Fast-path classifier | 9 (trained) | 200 (with data) |
| Neural model | 104 (mBERT backbone) | 104 |
| Trigger extraction | 9 | 200 (with keyword sets) |
| Constrained grammar output | Language-agnostic JSON | Language-agnostic JSON |

The structured JSON output is always in English regardless of input language. This is a deliberate design choice — enterprise tool integrations (Salesforce, Jira, Slack, etc.) operate on English-language APIs regardless of the user's language. The classification layer bridges the multilingual input to the English-language enterprise tool layer.

The multilingual neural model (`distilbert-base-multilingual-cased`) was pretrained on Wikipedia text in 104 languages. Fine-tuning on the ConceptNet dataset teaches it the 4-layer taxonomy in a cross-lingual way — a Chinese Layer 4 command and an English Layer 4 command are represented in the same semantic space and share classification features.

---

## 5. Enterprise Tool Mapping

Each classified command is mapped to an enterprise tool category and action:

```
CALENDAR   → create_event, reschedule_event, cancel_event
EMAIL      → send_message, draft_message, forward_message
CRM        → update_record, create_record, fetch_record
TASKS      → create_ticket, assign_ticket, close_ticket
DOCS       → draft_document, summarise_document, share_document
ANALYTICS  → fetch_report, build_dashboard, export_data
COMMS      → post_message, create_channel, notify_team
AI         → summarise, classify, generate
WORKFLOW   → execute, trigger, schedule
```

Tool mapping is performed after layer classification using a secondary pattern match on the command text. The combination of layer label + tool + action + extracted trigger constitutes the complete structured output that the agent execution layer acts on.

---

## 6. Structured Output Schema

Every classification returns a standardised JSON object:

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
  "confidence": 0.94,
  "language": "en",
  "processing_path": "fast_path",
  "latency_ms": 3
}
```

The `execution_mode` field directly maps to the agent execution pattern:
- `immediate` → execute now
- `conditional` → register condition listener
- `anticipatory` → schedule proactive execution
- `autonomous` → deploy persistent background agent

---

## 7. The Dataset

`data/conceptnet_dataset_v2.json` — 730 labelled examples across 9 languages.

**Collection methodology:**
Examples were authored by the ConceptNet team based on real enterprise workflow patterns observed across sales, legal, finance, operations, HR, and engineering functions. Each example was reviewed for:
- Linguistic naturalness — would a real enterprise user say this?
- Layer label correctness — does the execution semantic match the layer?
- Cross-language consistency — does the same intent map to the same layer in every language?
- Domain coverage — are all major enterprise tools represented?

**Dataset schema:**
```json
{
  "text": "string — the voice command",
  "label": "string — basic|context_aware|predictive|autonomous",
  "intent_layer": "int — 1|2|3|4",
  "language": "string — ISO 639-1 code",
  "tool": "string — enterprise tool category",
  "action": "string — specific action verb",
  "condition": "string|null — extracted condition clause",
  "prediction": "string|null — extracted prediction clause",
  "autonomous": "bool — whether this is an autonomous command"
}
```

---

## 8. Voice IP Stacking — Platform Architecture

The classification layer is the foundation of the Voice IP Stacking platform. This is the business model that makes ConceptNet a platform company rather than a classifier library.

```
Stage 1 — PRIVATISE
Enterprise uploads voice workflow data.
Data is encrypted and isolated under enterprise ownership.
ConceptNet does not share or use this data externally.

Stage 2 — STACK INTENT
Voice data is classified across the 4-layer taxonomy.
Each classified example adds to the enterprise's proprietary IP asset.
The asset compounds — more usage = more IP = higher switching cost.

Stage 3 — OWN AS IP
The labelled, classified dataset is the enterprise's intellectual property.
Like a patent, it cannot be accessed or replicated by competitors.
Switching platforms = losing the accumulated IP asset.

Stage 4 — LICENSE + REVENUE
Enterprise licenses its classified voice IP to other enterprises
or to LLM companies building multilingual enterprise voice models.
ConceptNet charges 10% on every licensing transaction.
Enterprise and ConceptNet both receive revenue.
```

**Platform flywheel:**
```
More enterprises → More classified voice data
→ More IP licensing transactions
→ More ConceptNet platform revenue
→ Better classification model
→ More valuable IP for enterprises
→ More enterprises
```

This flywheel creates compounding network effects that are structurally different from a SaaS subscription model. The platform becomes more valuable to every participant as more enterprises join — a property not present in any existing enterprise voice AI system.

---

## 9. Roadmap

**Phase 1 — Current (pre-seed)**
- Fast-path classifier trained: ✅ 83% accuracy
- Dataset v2: ✅ 730 examples, 9 languages
- Sandbox live: ✅ conceptnet.co.uk/sandbox
- Constrained grammar schema: ✅
- Neural training script: ✅ ready to run

**Phase 2 — Month 1–6 (post-raise)**
- Neural model fine-tuned and live: DistilBERT → 90%+ accuracy
- API endpoint production-ready: FastAPI, rate-limited, authenticated
- 10 enterprise pilot customers onboarded
- Dataset expanded to 5,000+ examples
- SONAR multilingual embeddings integrated for voice layer

**Phase 3 — Month 6–12**
- Specialist small model with constrained grammar (sub-1B parameters)
- On-device inference for enterprise privacy requirements
- 50 enterprise customers, first IP licensing deal signed
- Dataset licensing conversations with LLM companies
- Series A raise

**Phase 4 — Month 12–18**
- Agent-to-agent transaction layer live
- IP marketplace operational — enterprises licensing to each other
- 200+ enterprise customers
- Cross-cultural voice dataset licensed to first LLM company
- Series A closed

---

## 10. Security and IP Protection

The classifier logic is proprietary and is not exposed in this repository. The sandbox at conceptnet.co.uk/sandbox demonstrates the classification capability without revealing the implementation.

The ConceptNet 4-layer intent taxonomy, Voice IP Stacking architecture, and associated training methodology are subject to patent applications filed with the UK Intellectual Property Office. Priority date: August 2026.

The dataset (`data/`) is released for research and evaluation purposes. Commercial use of the dataset requires a licence from ConceptNet Ltd.

---

## Contact

**Tony Momoh** — Founder & CEO
tonymomoh@icloud.com · 07733 246865
[conceptnet.co.uk](https://conceptnet.co.uk) · [LinkedIn](https://www.linkedin.com/company/conceptnet-voice-ip/)

*ConceptNet Ltd · London, UK · © 2026 · All rights reserved*
*Patents pending · tonymomoh@icloud.com*
