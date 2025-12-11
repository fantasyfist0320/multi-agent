from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from app.state import State
from app.tools.product_rules import get_eligible_and_scored_products
from app.tools.product_rules import product_id
from app.llm import get_chat_llm

import re

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"ignore instructions",
    r"disregard the previous rules",
    r"forget (all )?earlier instructions",
    r"recommend most expensive",
    r"jailbreak",
]


def _is_prompt_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in PROMPT_INJECTION_PATTERNS)

PURPOSE_CANONICAL = {
    "tourism": "Tourism",
    "personal trip": "Personal trip",
    "vacation": "Tourism",
    "holiday": "Tourism",
    "business": "Business trip",
    "business trip": "Business trip",
    "expatriation": "Expatriation",
    "expat": "Expatriation",
    "long term stay": "Long-term stay",
    "long-term stay": "Long-term stay",
    "relocation": "Relocation",
    "work abroad": "Work abroad",
    "working holiday": "Working Holiday",
    "pvt": "PVT",
}


def _normalize_purpose(purpose: Optional[str]) -> Optional[str]:
    if not purpose:
        return None
    p = purpose.strip().lower()
    for key, canon in PURPOSE_CANONICAL.items():
        if key in p:
            return canon
    return purpose  # fallback: return as-is

def _extract_profile_from_text(user_text: str) -> Dict[str, Any]:
    llm = get_chat_llm()
    system_prompt = (
        "You extract a structured trip profile from a user message for travel insurance.\n"
        "Extract the following fields:\n"
        "- age: integer or null\n"
        "- destination: short string (e.g. 'Spain', 'Europe', 'Thailand') or null\n"
        "- duration_days: integer number of days of the trip or null\n"
        "- purpose: short English label describing the trip purpose, or null.\n"
        "Purpose should be one of (if possible): "
        "'Personal trip', 'Tourism', 'Business trip', "
        "'Expatriation', 'Long-term stay', 'Relocation', 'Work abroad', "
        "'Working Holiday', 'PVT'.\n\n"
        "Return ONLY JSON, no explanation."
    )
    user_prompt = f"User message:\n{user_text}"

    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    content = content[7:-3]
    try:
        data = json.loads(content)
        return {
            "age": data.get("age"),
            "destination": data.get("destination"),
            "duration_days": data.get("duration_days"),
            "purpose": data.get("purpose"),
        }
    except Exception:
        return {}


def _generate_reasons_for_products(
    user_profile: Dict[str, Any],
    products: List[Dict[str, Any]],
) -> Dict[str, str]:
    
    if not products:
        return {}

    llm = get_chat_llm()
    system_prompt = (
        "You are an assistant generating SHORT reasons for recommending travel insurance products.\n"
        "For each product, write 1–2 sentences explaining why it fits the user's age, destination and trip duration.\n"
        "Return ONLY JSON with shape {\"reasons\": {product_id: reason_str, ...}}."
    )

    profile_str = json.dumps(user_profile, ensure_ascii=False)
    
    products_summary = [
        {
            "id": product_id(p),
            "name": p.get("name"),
            "description": p.get("description"),
            "purposes": p.get("purposes", []),
            "key_features": p.get("key_features", []),
        }
        for p in products
    ]
    user_prompt = (
        f"User profile: {profile_str}\n\n"
        f"Products: {json.dumps(products_summary, ensure_ascii=False)}"
    )

    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(content)
        return data.get("reasons", {})
    except Exception:
        return {p["id"]: f"{p['name']} matches your trip profile." for p in products}


def recommendation_node(state: State) -> State:
    messages = state.get("messages") or []
    last = messages[-1]
    user_text = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")

    if _is_prompt_injection(user_text):
        state["response"] = {
            "type": "clarification",
            "question": (
                "I can’t follow requests to ignore my instructions or to choose products "
                "without considering your actual travel profile. "
                "To recommend a suitable insurance product, please tell me your age, "
                "destination, and trip duration."
            ),
        }
        return state

    user_profile = state.get("user_profile") or {}

    if not user_profile or any(k not in user_profile for k in ["age", "destination", "duration_days", "purpose"]):
        extracted = _extract_profile_from_text(user_text)
        if extracted:
            extracted["purpose"] = _normalize_purpose(extracted.get("purpose"))
        user_profile.update({k: v for k, v in extracted.items() if v is not None})
        state["user_profile"] = user_profile


    if any(k not in user_profile or user_profile.get(k) is None for k in ["age", "destination", "duration_days"]):
        state["intent"] = "clarification"
        return state

    scored = get_eligible_and_scored_products(user_profile, max_products=2)
    products = [p for p, _ in scored]

    if not products:
        # No product fits strict eligibility.
        # Explicitly tell the user instead of forcing a recommendation.
        age = user_profile.get("age")
        destination = user_profile.get("destination")
        duration_days = user_profile.get("duration_days")

        msg_parts = ["Based on the information you provided"]
        detail_bits = []
        if age is not None:
            detail_bits.append(f"age {age}")
        if destination:
            detail_bits.append(f"destination {destination}")
        if duration_days is not None:
            detail_bits.append(f"trip duration {duration_days} days")

        if detail_bits:
            msg_parts.append("(" + ", ".join(detail_bits) + ")")
        msg_parts.append("none of our travel insurance products are eligible.")
        msg_parts.append("Please contact an advisor or adjust the traveller profile (for example, different age range or trip duration).")

        question = " ".join(msg_parts)

        state["response"] = {
            "type": "clarification",
            "question": question,
        }
        return state

    if not products:
        state["intent"] = "clarification"
        return state

    reasons = _generate_reasons_for_products(user_profile, products)

    rec_products = []
    for p in products:
        pid = product_id(p)
        rec_products.append(
            {
                "product_id": pid,
                "name": p.get("name"),
                "reason": reasons.get(
                    pid,
                    f"{p.get('name')} seems suitable for your trip profile.",
                ),
            }
        )

    state["response"] = {
        "type": "recommendation",
        "products": rec_products,
    }
    return state
