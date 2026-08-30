from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os

app = FastAPI(title="ConceptNet API", version="1.0")

# Load model from HuggingFace
print("Loading ConceptNet model...")
classifier = pipeline(
    "text-classification",
    model="conceptnetUk/intent-classifier"
)
print("Model loaded!")

LABEL_MAP = {
    "Basic": {
        "layer": 1,
        "execution_mode": "immediate",
        "description": "Execute now"
    },
    "Context-Aware": {
        "layer": 2,
        "execution_mode": "conditional",
        "description": "Wait for trigger then act"
    },
    "Predictive": {
        "layer": 3,
        "execution_mode": "anticipatory",
        "description": "Act proactively before event"
    },
    "Autonomous": {
        "layer": 4,
        "execution_mode": "persistent",
        "description": "Run autonomously forever"
    }
}

# API keys
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
def classify(
    request: ClassifyRequest,
    x_api_key: str = Header(None)
):
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Contact tonymomoh@icloud.com for access."
        )

    result = classifier(request.text)[0]
    label = result["label"]
    score = result["score"]
    meta = LABEL_MAP.get(label, LABEL_MAP["Basic"])

    return ClassifyResponse(
        text=request.text,
        intent_layer=meta["layer"],
        intent_label=label,
        execution_mode=meta["execution_mode"],
        description=meta["description"],
        confidence=round(score, 4),
        language=request.language
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
