from __future__ import annotations

from fastapi import FastAPI

def create_api() -> FastAPI:
    app = FastAPI(title="ai-review-insight")
    return app