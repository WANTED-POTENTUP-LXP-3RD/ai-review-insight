from __future__ import annotations

from pydantic import BaseModel, Field


# -------------------------
# Inputs
# -------------------------
class ReviewItem(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(..., min_length=1)


# -------------------------
# Review Analyze (summary)
# -------------------------
class ReviewAnalyzeRequest(BaseModel):
    reviews: list[ReviewItem] = Field(default_factory=list)


class ReviewAnalyzeResponse(BaseModel):
    mood: str  # POSITIVE/NEUTRAL/NEGATIVE
    insightSummary: str
