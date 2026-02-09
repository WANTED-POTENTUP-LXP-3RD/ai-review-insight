from __future__ import annotations

from fastapi import FastAPI
from app.api.api import api_router


def create_app() -> FastAPI:
    app = FastAPI(title="ai-review-insight", version="0.1.0")
    app.include_router(api_router)

    @app.get("/health")
    def health():
        return {"status": "OK"}

    return app


app = create_app()