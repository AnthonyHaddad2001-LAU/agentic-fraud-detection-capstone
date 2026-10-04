"""
agents.py Defines individual specialized agents and their operational logic.
"""
from langchain_ollama import ChatOllama
from State import FraudDetectionState
from mcp_tools import tool_transaction_fetcher, tool_location_validator, tool_fraud_scoring_model

# Initialize local LLM instance via Ollama
llm = ChatOllama(model="llama3.1", temperature=0.0)


def ingestion_agent(state: FraudDetectionState) -> dict:
    """Monitors incoming transaction feeds and computes baseline variance."""
    trace = state.get("reasoning_trace", [])
    trace.append("Ingestion Agent: Checking credit limit and computing velocity baseline...")

    user_data = tool_transaction_fetcher(state["user_id"])
    flags = []

    # Rule Check 1: Limit Violation Check
    if (user_data["current_balance"] + state["amount"]) > user_data["monthly_credit_limit"]:
        flags.append("CREDIT_LIMIT_EXCEEDED")

    # Rule Check 2: Spike relative to normal spending
    if state["amount"] > (user_data["avg_transaction_amount"] * 3.0):
        flags.append("HIGH_VALUE_SPIKE")

    return {
        "anomaly_flags": flags,
        "user_spending_baseline": user_data,
        "reasoning_trace": trace
    }


def risk_agent(state: FraudDetectionState) -> dict:
    """Performs deep contextual risk verification and runs ML scoring models."""
    trace = state.get("reasoning_trace", [])
    trace.append("Risk Agent: Performing location checks and running ML risk assessment...")

    baseline = state["user_spending_baseline"]
    home_coords = baseline["home_location_coords"]

    # Check physical distance impossibility (assumed 0.2 hours time delta for demo)
    location_res = tool_location_validator(home_coords, state["location_coords"], time_delta_hours=0.2)

    flags = state["anomaly_flags"]
    if location_res["impossible_travel"]:
        flags.append("GEOGRAPHIC_IMPOSSIBILITY")

    # Compute probability score
    score = tool_fraud_scoring_model(
        amount=state["amount"],
        baseline_avg=baseline["avg_transaction_amount"],
        impossible_travel=location_res["impossible_travel"]
    )

    return {
        "anomaly_flags": flags,
        "ml_risk_score": score,
        "reasoning_trace": trace
    }


def action_agent(state: FraudDetectionState) -> dict:
    """Executes operational banking controls based on risk score."""
    trace = state.get("reasoning_trace", [])
    trace.append("Action Agent: Evaluating operational response thresholds...")

    score = state.get("ml_risk_score", 0.0)
    flags = state.get("anomaly_flags", [])

    if score >= 0.75:
        action = "BLOCK_CARD_TEMPORARY"
        reason = f"High risk score ({score}) combined with critical anomalies: {flags}."
    elif score >= 0.40 or "CREDIT_LIMIT_EXCEEDED" in flags:
        action = "TRIGGER_SMS_OTP_VERIFICATION"
        reason = f"Moderate risk detected ({score}). Account limit or location anomaly requires secondary authorization."
    else:
        action = "APPROVE_TRANSACTION"
        reason = "Transaction within safe variance thresholds."

    return {
        "action_taken": action,
        "action_reasoning": reason,
        "reasoning_trace": trace
    }


def audit_agent(state: FraudDetectionState) -> dict:
    """Generates structured, human-readable case reports for audit logs."""
    trace = state.get("reasoning_trace", [])
    trace.append("Audit Agent: Formulating final audit trail summary...")

    prompt = f"""
    You are an AI Compliance Auditor. Synthesize the following incident trace into an executive summary:
    - Transaction ID: {state['transaction_id']}
    - User ID: {state['user_id']}
    - Amount: ${state['amount']}
    - Anomaly Flags: {state['anomaly_flags']}
    - Calculated ML Risk Score: {state['ml_risk_score']}
    - Final Action Taken: {state['action_taken']}
    - Reasoning: {state['action_reasoning']}

    Output a concise, 3-sentence compliance summary explaining the rationale.
    """

    response = llm.invoke(prompt)

    return {
        "audit_summary": response.content.strip(),
        "reasoning_trace": trace
    }