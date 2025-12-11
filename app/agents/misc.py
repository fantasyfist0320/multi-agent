from __future__ import annotations

from app.agents.router import _is_ambiguous_policy_question
from app.state import State


from langchain_core.messages import BaseMessage

def clarification_node(state: State) -> State:
    user_profile = state.get("user_profile") or {}
    messages = state.get("messages") or []
    last_text = ""
    if messages:
        last = messages[-1]
        last_text = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")

    if _is_ambiguous_policy_question(last_text) and not user_profile:
        question = (
            "When you ask about what is covered, are you referring to a specific product such as "
            "EUROPAX, Globe Traveller, or ACS Expat? "
            "If you’re not sure which product fits your situation, I can first recommend the most "
            "suitable product for your trip and then detail what it covers."
        )
        state["response"] = {
            "type": "clarification",
            "question": question,
        }
        return state

    required_fields = ["age", "destination", "duration_days", "purpose"]
    missing = [f for f in required_fields if user_profile.get(f) is None]

    if missing:
        labels = {
            "age": "age",
            "destination": "destination (country or region)",
            "duration_days": "trip duration (in days)",
            "purpose": "trip purpose (tourism, business trip, expatriation, etc.)",
        }
        missing_labels = [labels.get(f, f) for f in missing]
        question = (
            "To recommend a suitable travel insurance product, I need a bit more information: "
            + ", ".join(missing_labels)
            + "."
        )
    else:
        question = (
            "Are you looking for a product recommendation (which insurance to choose for your trip) "
            "or a detailed explanation of what is covered by a specific product?"
        )

    state["response"] = {
        "type": "clarification",
        "question": question,
    }
    return state

def low_confidence_node(state: State) -> State:
    
    question = state.get("rag_query") or "your question"

    state["response"] = {
        "type": "policy_answer",
        "answer": (
            "Based on the policy documents I have, I’m not confident enough to give a reliable answer "
            f"to '{question}'. Please contact the insurer or refer to the full policy document for confirmation."
        ),
        "confidence": 0.0,
        "sources": [],
    }
    return state
