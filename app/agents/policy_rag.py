# app/agents/policy_rag.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from app.state import State
from app.tools.policy_retriever import retrieve_policy_chunks, PolicyChunk
from app.llm import get_chat_llm


CONFIDENCE_THRESHOLD = 0.4  # tune as needed


def _compute_confidence(chunks: List[PolicyChunk]) -> float:
    if not chunks:
        return 0.0
    max_score = max(c.score for c in chunks)
    # simple normalization; you can tune based on empirical distances
    return float(max_score)


def _generate_policy_answer(
    question: str,
    chunks: List[PolicyChunk],
) -> Dict[str, Any]:
    """
    Ask LLM to answer in English + provide sources and confidence.
    Returns dict with keys: answer, confidence, sources[].
    """
    llm = get_chat_llm()

    context_pieces = []
    for c in chunks:
        context_pieces.append(
            f"[{c.product} | {c.section}] {c.content}"
        )
    context_text = "\n\n".join(context_pieces)

    system_prompt = (
        "You are a travel insurance expert answering questions based ONLY on the provided policy excerpts.\n"
        "Instructions:\n"
        "- Answer in English.\n"
        "- Explicitly mention the product names (e.g. 'According to EUROPAX...').\n"
        "- If multiple products differ, explain the difference.\n"
        "- If the answer is not clearly supported by the excerpts, say you are not sure.\n"
        "- Return ONLY JSON with keys: answer (string), confidence (0.0–1.0),\n"
        "  sources (list of {product: str, section: str})."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Policy excerpts:\n{context_text}"
    )

    resp = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(content)
        answer = str(data.get("answer", ""))
        confidence = float(data.get("confidence", 0.0))
        sources = data.get("sources", [])
        # Basic shape normalization
        norm_sources = []
        for s in sources:
            norm_sources.append(
                {
                    "product": s.get("product", "UNKNOWN"),
                    "section": s.get("section") or "",
                }
            )
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": norm_sources,
        }
    except Exception:
        # Fallback: generic low-confidence message
        return {
            "answer": "I am not confident enough to answer this question based on the available policy excerpts.",
            "confidence": 0.0,
            "sources": [],
        }


def policy_rag_node(state: State) -> State:
    
    messages = state.get("messages") or []
    last = messages[-1]
    question = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")

    state["rag_query"] = question

    chunks = retrieve_policy_chunks(question, top_k=5)

    confidence = _compute_confidence(chunks)
    state["rag_confidence"] = confidence

    if confidence < CONFIDENCE_THRESHOLD or not chunks:
        return state

    rag_answer = _generate_policy_answer(question, chunks)

    final_conf = float((confidence + rag_answer.get("confidence", 0.0)) / 2.0)

    state["response"] = {
        "type": "policy_answer",
        "answer": rag_answer["answer"],
        "confidence": final_conf,
        "sources": rag_answer["sources"],
    }
    state["rag_confidence"] = final_conf
    return state
