from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings


from app.llm import get_chat_llm
from langchain_core.messages import SystemMessage, HumanMessage
import json

PRODUCTS_PATH = Path(__file__).resolve().parent.parent / "data" / "products.json"

@lru_cache()
def load_products() -> List[Dict[str, Any]]:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"products.json not found at {PRODUCTS_PATH}")
    with PRODUCTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def product_id(product: Dict[str, Any]) -> str:
    
    name = str(product.get("name", "")).strip().lower()
    return name.replace(" ", "_")


def _check_age(product: Dict[str, Any], age: Optional[int]) -> bool:
    if age is None:
        return False
    min_age = product.get("age_min")
    max_age = product.get("age_max")

    if min_age is not None and age < min_age:
        return False
    if max_age is not None and age > max_age:
        return False

    return True
    
def _check_purpose(product: Dict[str, Any], purpose: Optional[str]) -> bool:
    if not purpose:
        return False
    purpose_norm = purpose.strip().lower()
    allowed = product.get("purposes") or []
    allowed_norm = [str(p).strip().lower() for p in allowed]

    if any(purpose_norm == a or purpose_norm in a or a in purpose_norm for a in allowed_norm):
        return True

    return False



def llm_is_destination_covered(destination: str, allowed: list[str]) -> bool:
    llm = get_chat_llm()

    system_prompt = (
        "You are checking geographic coverage for travel insurance.\n"
        "Given a destination and a list of allowed destinations/regions,\n"
        "determine if the user's destination is INCLUDED.\n"
        "Return ONLY JSON: {\"covered\": true/false}.\n"
        "Think in terms of geography: e.g. France is in Europe, Thailand is in Asia.\n"
        "'Monde entier' / 'World' covers all countries."
    )

    user_prompt = json.dumps(
        {
            "destination": destination,
            "allowed_destinations": allowed,
        },
        ensure_ascii=False,
    )

    resp = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(content)
        return bool(data.get("covered", False))
    except Exception:
        return False


def _check_destination(product: Dict[str, Any], destination: Optional[str]) -> bool:
    if destination is None:
        return False

    allowed = product.get("destinations") or []

    return llm_is_destination_covered(destination, allowed)


def _check_duration(product: Dict[str, Any], duration_days: Optional[int]) -> bool:
    if duration_days is None:
        return False
    
    min_d = product.get("duration_min_days") or 0
    max_d = product.get("duration_max_days") or 3650
    if min_d is not None and duration_days < min_d:
        return False
    if max_d is not None and duration_days > max_d:
        return False
    
    return True


def is_product_eligible(
    product: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> bool:
    age = user_profile.get("age")
    destination = user_profile.get("destination")
    duration_days = user_profile.get("duration_days")
    purpose = user_profile.get("purpose")

    return (
        _check_age(product, age)
        and _check_destination(product, destination)
        and _check_duration(product, duration_days)
        and _check_purpose(product, purpose)
    )


def score_product(
    product: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> float:
    score = 1.0

    duration_days = user_profile.get("duration_days")
    if isinstance(duration_days, int):
        duration_max = product.get("duration_max_days") or duration_days
        ratio = duration_days / max(1, duration_max)
        score -= abs(ratio - 0.8) * 0.2

    purpose = user_profile.get("purpose")
    if purpose:
        purpose_norm = purpose.strip().lower()
        allowed = [str(p).strip().lower() for p in product.get("purposes", [])]
        if any(purpose_norm == a for a in allowed):
            score += 0.1

    return float(score)



def get_eligible_and_scored_products(
    user_profile: Dict[str, Any],
    max_products: int = 2,
) -> List[Tuple[Dict[str, Any], float]]:
    
    products = load_products()
    products = products.get("products")
    
    eligible: List[Tuple[Dict[str, Any], float]] = []

    for product in products:
        if is_product_eligible(product, user_profile):
            s = score_product(product, user_profile)
            eligible.append((product, s))

    eligible.sort(key=lambda ps: ps[1], reverse=True)
    return eligible[:max_products]
