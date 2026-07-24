# ConceptNet — Proprietary Voice IP for the Next Generation of AI

> Low-cost SaaS is the stepping stone. Proprietary voice IP is the moat.

[![Star on GitHub](https://img.shields.io/github/stars/wushu75/ConceptNet?style=social)](https://github.com/wushu75/ConceptNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-cyan.svg)](https://python.org)
[![Live Demo](https://img.shields.io/badge/demo-live-green)](https://demo.conceptnet.co.uk)
[![Benchmarks](https://img.shields.io/badge/benchmarks-live-orange)](https://conceptnet.co.uk/benchmarks/)

**Raising £500K Pre-Seed · EIS Eligible · London, UK**

[🌐 Website](https://conceptnet.co.uk) · [🎮 Live Demo](https://demo.conceptnet.co.uk) · [🧪 Sandbox](https://conceptnet.co.uk/sandbox/) · [📊 Benchmarks](https://conceptnet.co.uk/benchmarks/) · [⭐ Star this repo](https://github.com/wushu75/ConceptNet)

---

## What is ConceptNet?

ConceptNet is a voice-native agent operating system for enterprise work.

**Layer 1 — Low-Cost SaaS (The Stepping Stone)**
Voice workflow automation for enterprise teams. 5× cheaper than Claude/OpenAI, token-free, 200 languages, self-prompting agent loops. Goal: collect voice data from 500+ enterprise teams.

**Layer 2 — Proprietary Voice IP (The Moat)**
The voice data collected from SaaS becomes intellectual property:
- Cross-cultural voice dataset (10M+ voice hours, 200 languages) — licensed to LLM companies
- Human-robot sync standard — new protocol for robot voice control
- Agent-to-agent transaction layer — infrastructure for the AI agent economy
- Enterprise voice IP stacking — enterprises privatize their voice + instructions, stack 4 intent layers, own their IP, license it to others (we charge 10% transaction fee)

**SaaS is not the product. It is the data collection mechanism.**

---

## Quick Start

```bash
git clone https://github.com/wushu75/ConceptNet.git
cd ConceptNet
python examples/quickstart.py
```

Or try a single command:

```bash
python core/intent_classifier.py "Schedule a board meeting when the Q3 results are ready"
```

**Output:**
```
Input:  Schedule a board meeting when the Q3 results are ready
Layer:  2 — Context-Aware
Action: schedule  →  Tool: CALENDAR
Condition: the q3 results are ready
Confidence: 78%
Workflow JSON:
{
  "version": "1.0",
  "tool": "CALENDAR",
  "action": "create_event",
  "parameters": { "raw_command": "...", "target": "board meeting" },
  "triggers": { "condition": "the q3 results are ready" },
  "execution_mode": "context_aware"
}
```

---

## Voice IP Stacking — The Platform Model

Most platforms collect your data. ConceptNet lets enterprises **own** their voice IP, then license it.

| Step | What happens |
|------|-------------|
| **① Privatize** | Enterprise uploads voice + instructions. We encrypt and isolate it. They own it. |
| **② Stack Intent** | 4 layers: Basic → Context-aware → Predictive → Autonomous |
| **③ Own as IP** | Voice + instructions = enterprise intellectual property (like patents, but for voice + intent) |
| **④ License + Revenue** | Enterprise licenses their IP to others. We charge 10% transaction fee. |

---

## 4 Intent Layers

```
Layer 1 — Basic        "Schedule a meeting"                      → CALENDAR.create_event
Layer 2 — Context      "Send report when deal closes"             → EMAIL.send_message + trigger
Layer 3 — Predictive   "Alert manager before contract expires"    → CALENDAR.set_reminder + prediction
Layer 4 — Autonomous   "Always update Salesforce after every call" → CRM.update_record + autonomous=True
```

Try them live in the **[Voice IP Sandbox →](https://conceptnet.co.uk/sandbox/)**

---

## Why This Moat Lasts 10 Years

| Feature | Moat Strength |
|---------|--------------|
| 5× cheaper than OpenAI | ❌ Copied in 6 months |
| Voice UI | ❌ Google/Microsoft ship it in 3 months |
| Cross-cultural voice dataset | ✅ We collected it. Competitors can't. |
| Human-robot sync standard | ✅ We define it. Others must follow. |
| Agent-to-agent protocol | ✅ We are the layer. Not a plugin. |
| Enterprise voice IP stacking | ✅ 500+ companies build IP on us. Switching = losing years of IP. |

**See live benchmarks → [benchmarks.conceptnet.co.uk](https://conceptnet.co.uk/benchmarks/)**

---

## API

Run locally:

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload
```

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Email the client when the proposal is approved"}'
```

Interactive API docs: `http://localhost:8000/docs`

---

## Financial Model

| Revenue Stream | Year 1 | Year 2 | Year 3 | Year 5 |
|----------------|--------|--------|--------|--------|
| SaaS | £100K | £500K | £1M | £5M |
| Enterprise IP Licensing (10%) | £0 | £500K | £2M | £10M |
| Agent-to-Agent Tx Fees | £0 | £100K | £500K | £5M |
| Dataset Licensing | £0 | £0 | £500K | £5M |
| **Total** | **£100K** | **£1.1M** | **£4M** | **£25M** |

---

## Repo Structure

```
ConceptNet/
├── core/
│   ├── __init__.py
│   └── intent_classifier.py   # Voice intent classification engine
├── api/
│   └── server.py              # FastAPI REST endpoint
├── examples/
│   └── quickstart.py          # CLI demo
├── benchmarks/
│   └── index.html             # Live benchmark page
├── sandbox/
│   └── index.html             # Interactive Voice IP Sandbox
├── docs/
│   └── star.html              # GitHub star campaign page
├── requirements.txt
└── README.md
```

---

## Roadmap

- **Month 0–6:** Deploy SaaS, onboard 500 enterprise teams, collect 1M+ voice hours
- **Month 6–12:** Cross-cultural dataset v1, first LLM licensing deal, enterprise IP stacking beta
- **Month 12–24:** 100 enterprises building voice IP, agent-to-agent layer live, $5M ARR
- **Month 24–36:** 500+ enterprises, 50+ verticals, £25M ARR

---

## Raising £500K Pre-Seed

Co-funded alongside Innovate UK AI & Data Economy grant application. **EIS eligible — 50% tax relief for UK investors.**

- **Pre-money valuation:** £3–5M
- **Use of funds:** Engineering (65%), cloud compute (20%), enterprise pilots (10%), ops (5%)
- **Contact:** tonymomoh@icloud.com · 07733 246865
- **Website:** [conceptnet.co.uk](https://conceptnet.co.uk)

---

## The Flywheel

```
SaaS → Data → IP → Enterprise Voice IP → Leverage → New Use Cases → More Users → Better IP → More Leverage
```

---

## ⭐ Star this repo

If you find this useful or believe in open-source voice AI infrastructure, a GitHub star helps more developers discover ConceptNet.

[**→ Star on GitHub**](https://github.com/wushu75/ConceptNet)

*Built in London. Open-source. Voice-native. 10-year moat.*
