from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class RecommendedProduct(BaseModel):
    name: str = Field(..., description="Human-readable product name")
    reason: str = Field(..., description="Short explanation why this product was recommended")


class RecommendationResponse(BaseModel):
    type: Literal["recommendation"] = "recommendation"
    products: List[RecommendedProduct] = Field(
        ..., description="One or more recommended products"
    )

class PolicySource(BaseModel):
    product: str = Field(
        ...,
        description="Product where this information was found (e.g. 'EUROPAX', 'GLOBE TRAVELLER')",
    )
    section: Optional[str] = Field(
        None,
        description="Optional section/chapter/paragraph identifier (e.g. 'Article 3 – Coverage')",
    )


class PolicyAnswerResponse(BaseModel):
    type: Literal["policy_answer"] = "policy_answer"
    answer: str = Field(
        ...,
        description="Natural language answer in English, mentioning product names explicitly",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the answer, 0.0–1.0",
    )
    sources: List[PolicySource] = Field(
        ...,
        description="Where in the policy docs the answer comes from",
    )

class ClarificationResponse(BaseModel):
    type: Literal["clarification"] = "clarification"
    question: str = Field(
        ...,
        description="Clarifying question to the user when intent/profile is ambiguous",
    )

class QueryResponse(BaseModel):
    type: Literal["recommendation", "policy_answer", "clarification"]
