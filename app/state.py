from typing import Annotated, Dict, List, Literal, Optional, Any
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps


Intent = Literal["product_recommendation", "policy_question", "clarification"]


class State(TypedDict, total=False):

    messages: Annotated[List[AnyMessage], add_messages]
    remaining_steps: RemainingSteps

    intent: Optional[Intent]
    router_confidence: Optional[float]

    user_profile: Dict[str, Any]

    rag_query: Optional[str]
    rag_results: Optional[List[Dict[str, Any]]]
    rag_confidence: Optional[float]

    response: Optional[Dict[str, Any]]

    error: Optional[str]


def make_initial_state(user_message: str, max_steps: int) -> State:
    """
    Create the initial State for a new /api/query call.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": user_message,
            }
        ],
    }
