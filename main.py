"""
main.pynb Executes end-to-end multi-agent scenario runs.
"""
import json
from supervisor import build_fraud_detection_graph

def run_demo():
    app = build_fraud_detection_graph()

    # Sample Test Scenario: High-value transaction in Tokyo shortly after Paris baseline
    initial_state = {
        "transaction_id": "TXN_98421",
        "user_id": "USR_4412",
        "amount": 1250.00,
        "merchant": "Tokyo Electronics Corp",
        "merchant_category": "ELECTRONICS",
        "location_coords": (35.6762, 139.6503),  # Tokyo, Japan
        "timestamp": "2026-10-04T14:30:00Z",
        "anomaly_flags": [],
        "user_spending_baseline": {},
        "ml_risk_score": None,
        "action_taken": None,
        "action_reasoning": None,
        "audit_summary": None,
        "next_step": "",
        "reasoning_trace": []
    }

    print("=== Starting Multi-Agent Transaction Processing Pipeline ===")
    final_state = app.invoke(initial_state)

    print("\n--- Execution Reasoning Trace ---")
    for step in final_state["reasoning_trace"]:
        print(f"-> {step}")

    print("\n--- System Outcomes ---")
    print(f"Anomaly Flags    : {final_state['anomaly_flags']}")
    print(f"ML Risk Score    : {final_state['ml_risk_score']}")
    print(f"Action Executed  : {final_state['action_taken']}")
    print(f"Action Rationale : {final_state['action_reasoning']}")

    print("\n--- Final Audit Log ---")
    print(final_state["audit_summary"])

if __name__ == "__main__":
    run_demo()