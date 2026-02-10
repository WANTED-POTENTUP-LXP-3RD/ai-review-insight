from __future__ import annotations

from fastapi import APIRouter
from app.domains.review_insight.router import router as review_insight_router

api_router = APIRouter()
api_router.include_router(review_insight_router)