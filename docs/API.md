# ConceptNet API Documentation

## Overview

ConceptNet classifies enterprise voice and text commands into 4 intent layers — producing structured JSON for agent execution automatically.

**Base URL:**
```
https://api-inference.huggingface.co/models/conceptnetUk/intent-classifier
```

**Access:** Request your API token by emailing tonymomoh@icloud.com

---

## Authentication

Include your API token in every request header:

```
Authorization: Bearer YOUR_TOKEN_HERE
```

---

## Classify a Command

**POST** `https://api-inference.huggingface.co/models/conceptnetUk/intent-classifier`

### Request

```json
{
  "inputs": "Send the report when the contract is signed"
}
```

### Response

```json
[
  [
    {"label": "Context-Aware", "score": 0.9734},
    {"label": "Basic", "score": 0.0156},
    {"label": "Predictive", "score": 0.0072},
    {"label": "Autonomous", "score": 0.0038}
  ]
]
```

The highest scoring label is the classification. In this example: **Context-Aware (L2)**.

---

## The 4 Intent Layers

| Label | Layer | Pattern | Execution | Example |
|-------|-------|---------|-----------|---------|
| **Basic** | L1 | "Do X" | Immediate | "Schedule a meeting" |
| **Context-Aware** | L2 | "Do X when Y" | Wait for trigger | "Send report when contract is signed" |
| **Predictive** | L3 | "Do X before Y" | Proactive | "Alert manager before deadline" |
| **Autonomous** | L4 | "Do X always" | Persistent agent | "Auto-update CRM after every call" |

---

## Code Examples

### Python

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/conceptnetUk/intent-classifier"
TOKEN = "YOUR_TOKEN_HERE"

def classify(text):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"inputs": text}
    )
    results = response.json()[0]
    best = max(results, key=lambda x: x["score"])
    return {
        "label": best["label"],
        "confidence": round(best["score"], 4)
    }

# Examples
print(classify("Schedule a board meeting"))
# {"label": "Basic", "confidence": 0.9821}

print(classify("Send the report when the contract is signed"))
# {"label": "Context-Aware", "confidence": 0.9734}

print(classify("Alert the manager before the deadline expires"))
# {"label": "Predictive", "confidence": 0.9812}

print(classify("Automatically update Salesforce after every call"))
# {"label": "Autonomous", "confidence": 0.9756}
```

---

### JavaScript / Node.js

```javascript
const API_URL = "https://api-inference.huggingface.co/models/conceptnetUk/intent-classifier";
const TOKEN = "YOUR_TOKEN_HERE";

async function classify(text) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${TOKEN}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ inputs: text })
  });
  const data = await response.json();
  const results = data[0];
  const best = results.reduce((a, b) => a.score > b.score ? a : b);
  return {
    label: best.label,
    confidence: Math.round(best.score * 10000) / 10000
  };
}

// Example
classify("Send the report when the contract is signed")
  .then(result => console.log(result));
// {label: "Context-Aware", confidence: 0.9734}
```

---

### curl

```bash
curl -X POST \
  https://api-inference.huggingface.co/models/conceptnetUk/intent-classifier \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Alert the manager before the contract expires"}'
```


---

## Languages Supported

English · French · Spanish · German · Italian · Portuguese · Chinese · Arabic · Russian

Pass your text in any of these languages — the model classifies correctly across all 9.

```python
# French example
classify("Envoyer le rapport quand le contrat est signé")
# {"label": "Context-Aware", "confidence": 0.9612}

# Arabic example  
classify("تنبيه المدير قبل انتهاء العقد")
# {"label": "Predictive", "confidence": 0.9543}
```

---

## Performance

| Path | Accuracy | Latency | Cost |
|------|---------|---------|------|
| Fast-path | 83% | <5ms | Free |
| Neural (this API) | 100% | <500ms | Free |

---

## Rate Limits

| Tier | Requests | Cost |
|------|---------|------|
| Free (no token) | 30,000/month | £0 |
| With token | Higher limits | £0 |
| Production | Unlimited | Contact us |

---

## First Call — Model Warm Up

The first API call after a period of inactivity may take 20–30 seconds while the model loads. Subsequent calls are fast. This is normal behaviour on the free tier.

---

## Get Access

Email **tonymomoh@icloud.com** with:
- Your name and company
- What you're building
- Expected request volume

We'll send you a token within 24 hours.

---

## Links

- **Live sandbox:** https://conceptnet.co.uk/sandbox/
- **Model:** https://huggingface.co/conceptnetUk/intent-classifier
- **GitHub:** https://github.com/wushu75/ConceptNet
- **Website:** https://conceptnet.co.uk

*© 2026 ConceptNet Ltd · Patents pending · tonymomoh@icloud.com*
