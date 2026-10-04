"""
supervisor.py Defines the Supervisor Router pattern using LangGraph StateGraph.
"""
from langgraph.graph import StateGraph, END
from State import FraudDetectionState
from agents import ingestion_agent, risk_agent, action_agent, audit_agent


def supervisor_router(state: FraudDetectionState) -> str:
    """Determines next node in workflow execution sequence."""
    # Dynamic routing based on current state attributes
    if "user_spending_baseline" not in state or not state["user_spending_baseline"]:
        return "ingestion_node"
    elif state.get("ml_risk_score") is None:
        return "risk_node"
    elif state.get("action_taken") is None:
        return "action_node"
    elif state.get("audit_summary") is None:
        return "audit_node"
    else:
        return END


def build_fraud_detection_graph():
    """Constructs the LangGraph multi-agent execution pipeline."""
    workflow = StateGraph(FraudDetectionState)

    # Add Agent Nodes
    workflow.add_node("ingestion_node", ingestion_agent)
    workflow.add_node("risk_node", risk_agent)
    workflow.add_node("action_node", action_agent)
    workflow.add_node("audit_node", audit_agent)

    # Set Conditional Routing Entry & Transitions
    workflow.set_conditional_entry_point(
        supervisor_router,
        {
            "ingestion_node": "ingestion_node",
            "risk_node": "risk_node",
            "action_node": "action_node",
            "audit_node": "audit_node",
            END: END
        }
    )

    workflow.add_conditional_edges("ingestion_node", supervisor_router)
    workflow.add_conditional_edges("risk_node", supervisor_router)
    workflow.add_conditional_edges("action_node", supervisor_router)
    workflow.add_conditional_edges("audit_node", supervisor_router)

    return workflow.compile()