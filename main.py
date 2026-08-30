from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="ConceptNet API", version="1.0")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/conceptnetUk/intent-classifier"

LABEL_MAP = {
    "LABEL_0": {"layer": 1, "label": "Basic", "mode": "immediate", "description": "Execute now"},
    "LABEL_1": {"layer": 2, "label": "Context-Aware", "mode": "conditional", "description": "Wait for trigger then act"},
    "LABEL_2": {"layer": 3, "label": "Predictive", "mode": "anticipatory", "description": "Act proactively before event"},
    "LABEL_3": {"layer": 4, "label": "Autonomous", "mode": "persistent", "description": "Run autonomously forever"},
    "Basic": {"layer": 1, "label": "Basic", "mode": "immediate", "description": "Execute now"},
    "Context-Aware": {"layer": 2, "label": "Context-Aware", "mode": "conditional", "description": "Wait for trigger then act"},
    "Predictive": {"layer": 3, "label": "Predictive", "mode": "anticipatory", "description": "Act proactively before event"},
    "Autonomous": {"layer": 4, "label": "Autonomous", "mode": "persistent", "description": "Run autonomously forever"},
}

API_KEYS = {
    os.environ.get("API_KEY_1", "conceptnet_beta_2026"): "Beta User",
    os.environ.get("API_KEY_2", "conceptnet_internal"): "Internal",
}

class ClassifyRequest(BaseModel):
    text: str
    language: str = "en"

class ClassifyResponse(BaseModel):
    text: str
    intent_layer: int
    intent_label: str
    execution_mode: str
    description: str
    confidence: float
    language: str
    model: str = "conceptnetUk/intent-classifier"

@app.get("/")
def root():
    return {
        "name": "ConceptNet API",
        "version": "1.0",
        "status": "live",
        "docs": "/docs",
        "sandbox": "https://conceptnet.co.uk/sandbox/"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    x_api_key: str = Header(None)
):
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Contact tonymomoh@icloud.com for access."
        )

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            HF_API_URL,
            json={"inputs": request.text},
            headers=headers,
            timeout=30.0
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Model inference failed: {response.text}"
        )

    results = response.json()
    if isinstance(results, list) and len(results) > 0:
        if isinstance(results[0], list):
            results = results[0]
        best = max(results, key=lambda x: x["score"])
        label_key = best["label"]
        score = best["score"]
    else:
        raise HTTPException(status_code=502, detail="Unexpected model response")

    meta = LABEL_MAP.get(label_key, LABEL_MAP["Basic"])

    return ClassifyResponse(
        text=request.text,
        intent_layer=meta["layer"],
        intent_label=meta["label"],
        execution_mode=meta["mode"],
        description=meta["description"],
        confidence=round(score, 4),
        language=request.language
    )
