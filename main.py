from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.api_schemas import (
    RecommendationResponse,
    PolicyAnswerResponse,
    ClarificationResponse,
)
from app.state import make_initial_state
from app.config import get_settings
from app.graph import build_graph

from app.tools.policy_retriever import build_policy_index


app = FastAPI(title="Insurance Multi-Agent API")

settings = get_settings()
graph_app = build_graph()

class QueryIn(BaseModel):
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_policy_index(force_rebuild=False)
    yield


app = FastAPI(lifespan=lifespan)
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/api/query",
    responses={
        200: {
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            RecommendationResponse.model_json_schema(),
                            PolicyAnswerResponse.model_json_schema(),
                            ClarificationResponse.model_json_schema(),
                        ]
                    }
                }
            },
            "description": "Multi-agent response",
        }
    },
)
def query(payload: QueryIn):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    state = make_initial_state(payload.message, max_steps=settings.max_steps)

    final_state = graph_app.invoke(
        state,
        config={"configurable": {"thread_id": "api-session"}},
    )

    response = final_state.get("response")
    if response is None:
        raise HTTPException(
            status_code=500,
            detail="Agent graph finished without a response.",
        )
    return response