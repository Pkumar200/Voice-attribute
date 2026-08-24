"""
Pydantic models for the /analyze API contract.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

GenderLabel = Literal["male", "female", "unknown"]
AgeBracketLabel = Literal["18-30", "31-45", "46-60", "60+", "unknown"]
AudioQuality = Literal["good", "degraded", "insufficient"]


class AttributeScore(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class GenderScore(AttributeScore):
    prediction: GenderLabel


class AgeBracketScore(AttributeScore):
    prediction: AgeBracketLabel


class LanguageScore(BaseModel):
    prediction: Optional[str] = None  # e.g. "en", "es" (best-effort, bonus feature)
    confidence: float = 0.0


class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderScore
    age_bracket: AgeBracketScore
    language: Optional[LanguageScore] = None
    processing_ms: int
    audio_quality: AudioQuality
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    contact_id: Optional[str] = None
    error: str
    detail: Optional[str] = None
