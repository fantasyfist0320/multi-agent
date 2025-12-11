from pydantic import BaseModel
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str
    openai_model_chat: str = "gpt-4.1-mini"
    openai_model_embed: str = "text-embedding-3-small"

    vector_db_dir: str = ".vectorstore"

    max_steps: int = 8
    max_tokens_per_call: int = 4096


@lru_cache()
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_model_chat=os.environ.get("OPENAI_MODEL_CHAT", "gpt-4.1-mini"),
        openai_model_embed=os.environ.get("OPENAI_MODEL_EMBED", "text-embedding-3-small"),
        vector_db_dir=os.environ.get("VECTOR_DB_DIR", ".vectorstore"),
        max_steps=int(os.environ.get("MAX_STEPS", "8")),
        max_tokens_per_call=int(os.environ.get("MAX_TOKENS_PER_CALL", "4096")),
    )
