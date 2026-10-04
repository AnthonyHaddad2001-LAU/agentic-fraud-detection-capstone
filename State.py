"""
state.py Defines the shared memory/state object passed across the agent pipeline.
"""
from typing import TypedDict, List, Dict, Any, Optional

class FraudDetectionState(TypedDict):
    # Raw Input
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    merchant_category: str
    location_coords: tuple[float, float]
    timestamp: str

    # Internal Agent Outputs
    anomaly_flags: List[str]
    user_spending_baseline: Dict[str, Any]
    ml_risk_score: Optional[float]
    action_taken: Optional[str]
    action_reasoning: Optional[str]
    audit_summary: Optional[str]

    # Routing Control
    next_step: str
    reasoning_trace: List[str]