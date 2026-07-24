"""
ConceptNet Core — Intent Classifier
Classifies voice/text commands into structured enterprise workflow tasks.
Runs locally, no API key required.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class IntentResult:
    raw_input: str
    intent_layer: int          # 1=Basic, 2=Context, 3=Predictive, 4=Autonomous
    intent_label: str          # Human-readable layer name
    action: str                # Primary action verb
    target: str                # What the action acts on
    condition: Optional[str]   # "when Y" trigger (Layer 2+)
    prediction: Optional[str]  # "before Y" trigger (Layer 3+)
    confidence: float          # 0.0–1.0
    workflow_json: dict        # Structured output for agent execution


# Action verb patterns → enterprise tool mapping
ACTION_MAP = {
    r"\b(schedule|book|arrange|set up|create).*(meeting|call|event|appointment)\b": ("CALENDAR", "create_event"),
    r"\b(send|email|message|notify|ping|dm)\b": ("EMAIL", "send_message"),
    r"\b(update|log|record|note|add).*(crm|salesforce|hubspot|contact|lead|deal)\b": ("CRM", "update_record"),
    r"\b(create|open|add|make|file).*(ticket|issue|task|jira|bug|request)\b": ("TASKS", "create_ticket"),
    r"\b(summarise|summarize|recap|brief|tldr)\b": ("AI", "summarise"),
    r"\b(pull|fetch|get|retrieve|show).*(report|data|stats|numbers|pipeline|dashboard)\b": ("ANALYTICS", "fetch_report"),
    r"\b(post|share|publish|announce).*(slack|teams|channel)\b": ("COMMS", "post_message"),
    r"\b(draft|write|compose|prepare).*(email|proposal|doc|document|report)\b": ("DOCS", "draft_document"),
    r"\b(remind|reminder|alert|follow.?up)\b": ("CALENDAR", "set_reminder"),
    r"\b(approve|review|sign.?off|authorise|authorize)\b": ("WORKFLOW", "request_approval"),
}

# Condition patterns (Layer 2 — context-aware)
CONDITION_PATTERNS = [
    r"\bwhen\s+(.+?)(?:\s+and|\s+but|,|$)",
    r"\bif\s+(.+?)(?:\s+and|\s+but|,|$)",
    r"\bafter\s+(.+?)(?:\s+and|\s+but|,|$)",
    r"\bonce\s+(.+?)(?:\s+and|\s+but|,|$)",
]

# Prediction patterns (Layer 3 — predictive intent)
PREDICTION_PATTERNS = [
    r"\bbefore\s+(.+?)(?:\s+and|\s+but|,|$)",
    r"\bin advance of\s+(.+?)(?:\s+and|\s+but|,|$)",
    r"\bproactively\s+(.+?)(?:\s+and|\s+but|,|$)",
    r"\banticipate\s+(.+?)(?:\s+and|\s+but|,|$)",
]

# Autonomous indicators (Layer 4)
AUTONOMOUS_KEYWORDS = [
    "automatically", "autonomously", "on its own", "without asking",
    "without being told", "unprompted", "always", "every time",
    "whenever", "continuously", "keep doing", "auto-"
]


def classify(text: str) -> IntentResult:
    """Classify a voice or text command into a structured intent."""
    t = text.lower().strip()

    # Detect action + tool
    action, tool, tool_action = "process", "WORKFLOW", "execute"
    for pattern, (t_name, t_action) in ACTION_MAP.items():
        if re.search(pattern, t, re.IGNORECASE):
            action = re.search(r"\b(\w+)\b", pattern.replace(r"\b", "").split("|")[0]).group(1)
            tool, tool_action = t_name, t_action
            break

    # Extract target (nouns after action verb)
    target_match = re.search(r"(?:the |a |an )?([\w\s]{3,40}?)(?:\s+for|\s+to|\s+when|\s+if|$)", t)
    target = target_match.group(1).strip() if target_match else "task"

    # Layer 2 — condition detection
    condition = None
    for pat in CONDITION_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            condition = m.group(1).strip()
            break

    # Layer 3 — prediction detection
    prediction = None
    for pat in PREDICTION_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            prediction = m.group(1).strip()
            break

    # Layer 4 — autonomous detection
    is_autonomous = any(kw in t for kw in AUTONOMOUS_KEYWORDS)

    # Determine intent layer
    if is_autonomous:
        layer, label = 4, "Autonomous"
    elif prediction:
        layer, label = 3, "Predictive"
    elif condition:
        layer, label = 2, "Context-Aware"
    else:
        layer, label = 1, "Basic"

    # Confidence heuristic
    confidence = min(0.95, 0.60 + (layer * 0.08) + (0.05 if tool != "WORKFLOW" else 0))

    # Build structured workflow JSON for agent execution
    workflow = {
        "version": "1.0",
        "tool": tool,
        "action": tool_action,
        "parameters": {
            "raw_command": text,
            "target": target,
        },
        "triggers": {},
        "execution_mode": label.lower().replace("-", "_"),
    }
    if condition:
        workflow["triggers"]["condition"] = condition
    if prediction:
        workflow["triggers"]["prediction"] = prediction
    if is_autonomous:
        workflow["triggers"]["autonomous"] = True

    return IntentResult(
        raw_input=text,
        intent_layer=layer,
        intent_label=label,
        action=action,
        target=target,
        condition=condition,
        prediction=prediction,
        confidence=confidence,
        workflow_json=workflow,
    )


def classify_json(text: str) -> str:
    """Return JSON string of classification result."""
    return json.dumps(asdict(classify(text)), indent=2)


# ── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    examples = [
        "Schedule a board meeting for Tuesday",
        "Send the Q3 report to the leadership team when the data is ready",
        "Create a Jira ticket before the sprint planning session",
        "Automatically update the CRM whenever a deal closes",
    ]
    inputs = sys.argv[1:] if len(sys.argv) > 1 else examples
    for cmd in inputs:
        print(f"\n{'─'*60}")
        print(f"Input:  {cmd}")
        result = classify(cmd)
        print(f"Layer:  {result.intent_layer} — {result.intent_label}")
        print(f"Action: {result.action}  →  Tool: {result.workflow_json['tool']}")
        if result.condition:  print(f"Condition: {result.condition}")
        if result.prediction: print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Workflow JSON:\n{json.dumps(result.workflow_json, indent=2)}")
