from __future__ import annotations

from fastapi import FastAPI
from app.api.api import api_router

# FastAPI 앱을 생성하고(title/version 설정), 라우터를 붙이고,
# /health를 추가한 뒤, uvicorn이 인식할 app 객체를 노출
def create_app() -> FastAPI:
    app = FastAPI(title="ai-review-insight", version="0.1.0")
    app.include_router(api_router)

    @app.get("/health")
    def health():
        return {"status": "OK"}

    return app


app = create_app()