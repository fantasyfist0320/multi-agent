from __future__ import annotations

import json
import re
from typing import Literal

from langgraph.graph import END
from langchain_core.messages import HumanMessage, SystemMessage

from app.state import State, Intent
from app.llm import get_chat_llm


INTENT_TYPES: list[Intent] = [
    "product_recommendation",
    "policy_question",
    "clarification",
]

PRODUCT_NAME_PATTERNS = [
    r"\beuropax\b",
    r"\bglobe\b",
    r"\btraveller\b",
    r"\bacs\b",
    r"\bexpat\b",
]



SPECIFIC_COVERAGE_KEYWORDS = [
    "pre-existing",
    "preexisting",
    "condition",
    "conditions",
    "repatriation",
    "bagage",
    "baggage",
    "medical",
    "hospital",
    "hospitalization",
    "cancel",
    "cancellation",
    "franchise",
    "deductible",
    "refund",
    "delay",
    "accident",
    "death",
    "liability",
]


def _mentions_product_name(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in PRODUCT_NAME_PATTERNS)

def _is_ambiguous_policy_question(text: str) -> bool:
    t = text.lower().strip()
    tokens = t.split()

    if "cover" not in t and "coverage" not in t:
        return False

    if _mentions_product_name(t):
        return False

    if any(k in t for k in SPECIFIC_COVERAGE_KEYWORDS):
        return False

    if len(tokens) <= 5:
        return True

    return False

def _heuristic_intent(user_text: str) -> tuple[Intent | None, float]:
    text = user_text.lower()

    policy_keywords = [
        "coverage",
        "covered",
        "cover",
        "refund",
        "reimbursement",
        "what is covered",
        "franchise",
        "deductible",
        "claim",
        "limit",
        "limits",
        "exclusion",
    ]
    reco_keywords = [
        "recommend",
        "which insurance",
        "which plan",
        "what insurance",
        "best product",
        "which package",
    ]

    has_age = bool(re.search(r"\b\d{2}\s*(years old|yo)\b", text))
    has_destination = any(k in text for k in ["trip to", "going to", "travel to", "vacation in"])

    if any(k in text for k in policy_keywords):
        if _is_ambiguous_policy_question(user_text):
            return "clarification", 0.8
        return "policy_question", 0.85

    if any(k in text for k in reco_keywords) and (has_age or has_destination):
        return "product_recommendation", 0.8

    return None, 0.0


def _llm_classify_intent(user_text: str) -> tuple[Intent, float]:

    llm = get_chat_llm()

    system_prompt = (
        "You are an intent classifier for a travel insurance assistant.\n"
        "You must classify the user's message into one of:\n"
        "1) product_recommendation – user wants a suggestion of which insurance product to buy.\n"
        "2) policy_question – user asks detailed questions about what is covered, limits, claims, etc.\n"
        "3) clarification – the message is too vague or you need more info to know what they want.\n\n"
        "Return ONLY a JSON object with keys: intent (string) and confidence (number between 0 and 1).\n"
        "Example: {\"intent\": \"policy_question\", \"confidence\": 0.78}"
    )
    user_prompt = f"User message:\n{user_text}"

    resp = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(content)
        intent = data.get("intent")
        conf = float(data.get("confidence", 0.0))
        if intent not in INTENT_TYPES:
            raise ValueError("Invalid intent")
        return intent, conf
    except Exception:
        return "clarification", 0.5


def router_node(state: State) -> State:
    
    messages = state.get("messages") or []
    if not messages:
        state["intent"] = "clarification"
        state["router_confidence"] = 0.0
        return state

    last = messages[-1]
    user_text = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")

    intent, conf = _heuristic_intent(user_text)

    if intent is None or conf < 0.7:
        intent, conf = _llm_classify_intent(user_text)

    state["intent"] = intent
    state["router_confidence"] = conf
    return state


def route_selector(state: State) -> str:
    """
    For LangGraph conditional edges: map state -> next node name.
    """
    intent = state.get("intent")
    if intent == "product_recommendation":
        return "recommendation"
    if intent == "policy_question":
        return "policy_rag"
    
    return "clarification"
