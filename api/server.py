"""
ConceptNet API — Voice Intent Classification Endpoint
Run: uvicorn api.server:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.intent_classifier import classify, IntentResult

app = FastAPI(
    title="ConceptNet API",
    description="Voice-native intent classification for enterprise workflow automation.",
    version="0.1.0",
    contact={"name": "Tony Momoh", "email": "tonymomoh@icloud.com", "url": "https://conceptnet.co.uk"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Schedule a board meeting for Tuesday when the quarterly results are ready",
                "language": "en",
            }
        }


class ClassifyResponse(BaseModel):
    input: str
    intent_layer: int
    intent_label: str
    action: str
    target: str
    condition: Optional[str]
    prediction: Optional[str]
    confidence: float
    workflow_json: dict
    latency_ms: float
    model: str = "conceptnet-intent-v0.1"


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "ConceptNet API",
        "status": "live",
        "version": "0.1.0",
        "docs": "/docs",
        "github": "https://github.com/wushu75/ConceptNet",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse, tags=["Intent"])
def classify_intent(req: ClassifyRequest):
    """
    Classify a voice or text command into a structured enterprise workflow intent.

    Returns one of 4 intent layers:
    - **Layer 1 — Basic**: Simple command ("Do X")
    - **Layer 2 — Context-Aware**: Conditional ("Do X when Y")
    - **Layer 3 — Predictive**: Anticipatory ("Do X before Y")
    - **Layer 4 — Autonomous**: Self-directed ("Do X automatically")
    """
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters.")
    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text must be under 2000 characters.")

    t0 = time.perf_counter()
    result = classify(req.text)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    return ClassifyResponse(
        input=result.raw_input,
        intent_layer=result.intent_layer,
        intent_label=result.intent_label,
        action=result.action,
        target=result.target,
        condition=result.condition,
        prediction=result.prediction,
        confidence=result.confidence,
        workflow_json=result.workflow_json,
        latency_ms=latency,
    )


@app.post("/classify/batch", tags=["Intent"])
def classify_batch(texts: list[str]):
    """Classify up to 50 commands in one call."""
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 inputs per batch.")
    return [classify(t).__dict__ for t in texts]
