from __future__ import annotations

from fastapi import APIRouter

from app.domains.review_insight.schema import (
    ReviewAnalyzeRequest,
    ReviewAnalyzeResponse,
)
from app.domains.review_insight.service import ReviewInsightService

router = APIRouter(tags=["review-insight"])

_service = ReviewInsightService()


@router.post("/internal/review/analyze", response_model=ReviewAnalyzeResponse)
async def analyze(req: ReviewAnalyzeRequest) -> ReviewAnalyzeResponse:
    return await _service.analyze(req.reviews)