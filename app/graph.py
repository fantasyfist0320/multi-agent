from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import State
from app.config import get_settings
from app.agents.router import router_node, route_selector
from app.agents.recommendation import recommendation_node
from app.agents.policy_rag import policy_rag_node
from app.agents.misc import clarification_node, low_confidence_node


def router_branch(state: State) -> str:
    
    return route_selector(state)


def recommendation_branch(state: State) -> str:
    
    if state.get("response"):
        return "__end__"
    
    if state.get("intent") == "clarification":
        return "clarification"
    return "__end__"


def policy_rag_branch(state: State) -> str:
    
    rag_conf = float(state.get("rag_confidence") or 0.0)
    if state.get("response") and rag_conf >= 0.0:
        return "__end__"
    return "low_confidence"

def build_graph():
    workflow = StateGraph(State)

    # --- Nodes ---
    workflow.add_node("router", router_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("policy_rag", policy_rag_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("low_confidence", low_confidence_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        router_branch,
        {
            "recommendation": "recommendation",
            "policy_rag": "policy_rag",
            "clarification": "clarification",
        },
    )
    workflow.add_conditional_edges(
        "recommendation",
        recommendation_branch,
        {
            "clarification": "clarification",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "policy_rag",
        policy_rag_branch,
        {
            "low_confidence": "low_confidence",
            "__end__": END,
        },
    )

    workflow.add_edge("clarification", END)

    workflow.add_edge("low_confidence", END)

    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app
