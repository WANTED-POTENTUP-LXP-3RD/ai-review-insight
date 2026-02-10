from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class Settings:
    # Java 내부망 Base URL # Java로 콜백 준비 (현재 사용 되지 않음)
    # JAVA_INTERNAL_BASE_URL: str = os.getenv("JAVA_INTERNAL_BASE_URL", "http://localhost:8080")

    # http client timeout (seconds)
    HTTP_CONNECT_TIMEOUT: float = _float("HTTP_CONNECT_TIMEOUT", 2.0)
    HTTP_READ_TIMEOUT: float = _float("HTTP_READ_TIMEOUT", 5.0)

    # 입력 상한 (MVP 안전장치)
    MAX_REVIEWS: int = _int("MAX_REVIEWS", 50)
    MAX_COMMENT_LEN: int = _int("MAX_COMMENT_LEN", 500)

    # 로깅 레벨 등은 필요 시 확장


settings = Settings()
