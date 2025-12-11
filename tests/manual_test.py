# from app.graph import build_graph
# from app.state import make_initial_state
# from app.config import get_settings
# from app.tools.policy_retriever import build_policy_index



# if __name__ == "__main__":
#     build_policy_index(force_rebuild=False)
#     run_query("I'm going to Spain for 2 months, I'm 35 years old, it's for tourism")



from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.graph import build_graph
from app.state import make_initial_state
from app.config import get_settings
from app.tools.policy_retriever import build_policy_index

ROOT = Path(__file__).resolve().parent
QUERIES_PATH = ROOT / "test_queries.json"


def load_queries() -> List[Dict[str, Any]]:
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_queries"]


def run_query(msg: str):
    settings = get_settings()
    graph_app = build_graph()
    state = make_initial_state(msg, max_steps=settings.max_steps)
    final = graph_app.invoke(
        state,
        config={"configurable": {"thread_id": "manual-test"}},
    )

    print("Final state:", final.get("response"))

    return final.get("response")


def behavior_check(query_def: Dict[str, Any], response: Dict[str, Any]) -> bool:
    """
    Very lightweight behavior checks based on test id.
    You can tighten this later.
    """
    qid = query_def["id"]
    expected_type = query_def["expected_type"]
    rtype = response.get("type")

    # First, basic type check (with some leniency for 9 and 10 if you return clarification)
    type_ok = (rtype == expected_type)
    if qid in (9, 10) and rtype == "clarification":
        type_ok = True  # we accept clarification as valid behavior

    # Per-id behavior checks
    if expected_type == "recommendation":
        products = [p.get("product_id", "").lower() for p in response.get("products", [])]

        if qid in (1, 2):
            # Should recommend EUROPAX
            return type_ok and any("europax" in p for p in products)

        if qid == 3:
            # Should recommend ACS Expat (we look for "acs" and/or "expat" in id)
            return type_ok and any("acs" in p or "expat" in p for p in products)

        if qid == 9:
            # No eligible product: we expect a clarification-style message
            return rtype == "clarification"

        if qid == 10:
            # Prompt injection: we expect a clarification/refusal, not a blind product list
            return rtype == "clarification"

        # default: just having products is OK
        return type_ok and bool(products)

    if expected_type == "policy_answer":
        if rtype != "policy_answer":
            return False
        answer = response.get("answer", "")
        sources = response.get("sources", []) or []

        if not answer or not sources:
            return False

        if qid == 4:
            # EUROPAX should appear in sources and answer
            return any("EUROPAX" in s.get("product", "").upper() for s in sources) and (
                "EUROPAX" in answer.upper()
            )

        if qid == 5:
            # GLOBE TRAVELLER should appear in sources and answer
            return any("GLOBE" in s.get("product", "").upper() for s in sources) and (
                "GLOBE" in answer.upper()
            )

        if qid == 6:
            # At least one source and some mention of exclusion or not covered
            negative_keywords = ["not covered", "exclusion", "excluded", "préexistant", "pre-existing"]
            return any(k in answer.lower() for k in negative_keywords) and bool(sources)

        # generic policy answer: just require sources and non-empty answer
        return True

    if expected_type == "clarification":
        return rtype == "clarification" and bool(response.get("question"))

    # fallback
    return False


def run_eval():
    settings = get_settings()
    graph_app = build_graph()
    tests = load_queries()

    results = []
    passed = 0

    for t in tests:
        qid = t["id"]
        query = t["query"]

        response = run_query(query)

        rtype = response.get("type")
        expected_type = t["expected_type"]
        ok = behavior_check(t, response)
        if ok:
            passed += 1

        results.append(
            {
                "id": qid,
                "query": query,
                "expected_type": expected_type,
                "actual_type": rtype,
                "pass": ok,
                "response": response,
            }
        )

    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed:      {passed}")
    print("=" * 80)
    for r in results:
        status = "✅ PASS" if r["pass"] else "❌ FAIL"
        print(f"[{status}] id={r['id']} expected={r['expected_type']} got={r['actual_type']}")
        print(f"Query:    {r['query']}")
        print(f"Response: {json.dumps(r['response'], ensure_ascii=False)}")
        print("-" * 80)


if __name__ == "__main__":
    build_policy_index(force_rebuild=False)
    run_eval()
    # run_query("I'm going to Spain for 2 months, I'm 35 years old, it's for tourism")
