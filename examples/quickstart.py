"""
ConceptNet — Quickstart Examples
Run: python examples/quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.intent_classifier import classify

print("\n" + "=" * 65)
print("  ConceptNet — Voice Intent Classification")
print("  github.com/wushu75/ConceptNet")
print("=" * 65)

LAYER_COLOURS = {1: "\033[94m", 2: "\033[93m", 3: "\033[95m", 4: "\033[92m"}
RESET = "\033[0m"
BOLD  = "\033[1m"

examples = [
    # Layer 1 — Basic
    "Schedule a board meeting for Tuesday",
    "Send the weekly report to the team",
    "Create a Jira ticket for the login bug",

    # Layer 2 — Context-Aware
    "Email the client when the proposal is approved",
    "Update the CRM record if the deal closes",
    "Notify the team after the deployment finishes",

    # Layer 3 — Predictive
    "Prepare the board pack before the quarterly review",
    "Alert the account manager before the contract expires",

    # Layer 4 — Autonomous
    "Automatically log all customer calls to Salesforce",
    "Always send a follow-up email whenever a meeting ends",
]

for cmd in examples:
    r = classify(cmd)
    col = LAYER_COLOURS.get(r.intent_layer, "")
    print(f"\n{col}{BOLD}[Layer {r.intent_layer} — {r.intent_label}]{RESET}")
    print(f"  Input:  {cmd}")
    print(f"  Tool:   {r.workflow_json['tool']}  →  {r.workflow_json['action']}")
    if r.condition:  print(f"  When:   {r.condition}")
    if r.prediction: print(f"  Before: {r.prediction}")
    print(f"  Confidence: {r.confidence:.0%}")

print("\n" + "=" * 65)
print("  Try your own:")
print("  python examples/quickstart.py 'Send the P&L to finance when month closes'")
print("=" * 65 + "\n")

if len(sys.argv) > 1:
    custom = " ".join(sys.argv[1:])
    r = classify(custom)
    col = LAYER_COLOURS.get(r.intent_layer, "")
    print(f"\n{col}{BOLD}Your input — Layer {r.intent_layer} ({r.intent_label}){RESET}")
    print(f"  Tool: {r.workflow_json['tool']}  →  {r.workflow_json['action']}")
    if r.condition:  print(f"  Trigger: {r.condition}")
    if r.prediction: print(f"  Prediction: {r.prediction}")
    import json
    print(f"\n  Full workflow JSON:\n{json.dumps(r.workflow_json, indent=4)}\n")
