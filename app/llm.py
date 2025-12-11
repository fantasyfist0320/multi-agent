from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List
from app.config import get_settings

settings = get_settings()

def get_chat_llm():
    return ChatOpenAI(
        model=settings.openai_model_chat,
        temperature=0.1,
        max_tokens=settings.max_tokens_per_call,
        openai_api_key=settings.openai_api_key,
    )

def simple_chat_call(system_prompt: str, user_prompt: str) -> str:
    llm = get_chat_llm()
    resp = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return resp.content if isinstance(resp.content, str) else str(resp.content)
