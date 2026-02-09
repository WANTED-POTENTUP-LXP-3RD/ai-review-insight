from __future__ import annotations

import os
import re
from typing import Iterable, List

import httpx # python -m pip install httpx
from fastapi import HTTPException
from kiwipiepy import Kiwi  # type: ignore
from transformers import pipeline  # type: ignore

from app.core.config import settings
from app.domains.review_insight.schema import (
    ReviewItem,
    ReviewAnalyzeResponse,
)

# (선택) 로컬 개발에서 .env 쓰고 싶으면
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ============================================
# 0) 전역 리소스 (프로세스 1회)
# ============================================
_KIWI = Kiwi()
MODEL_NAME = os.getenv("HF_MODEL", "eenzeenee/t5-base-korean-summarization")
_SUMMARIZER = None


def get_summarizer():
    """HF summarization pipeline을 프로세스 내 1회만 생성해서 재사용"""
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline("summarization", model=MODEL_NAME)
    return _SUMMARIZER


# ============================================
# Mood (별점 기반)
# ============================================
def tone_from_rating(avg: float) -> str:
    if avg >= 4.0:
        return "POSITIVE"
    if avg <= 2.0:
        return "NEGATIVE"
    return "NEUTRAL"


def avg_rating(reviews: Iterable[ReviewItem]) -> float:
    rs = list(reviews)
    if not rs:
        return 0.0
    return sum(r.rating for r in rs) / len(rs)


def aggregate_tone(reviews: Iterable[ReviewItem]) -> str:
    rs = list(reviews)
    if not rs:
        return "NEUTRAL"
    avg = sum(r.rating for r in rs) / len(rs)
    return tone_from_rating(avg)


# ============================================
# 입력 전처리
# ============================================
def _sanitize_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`+", " ", t)
    t = t.replace("<s>", " ").replace("</s>", " ")
    t = t.replace("▁", " ")
    t = re.sub(r"[#]{2,}", " ", t)

    drop_prefixes = (
        "prompt:", "instruction:", "system:", "assistant:", "user:",
        "요약:", "요약해", "요약해줘", "요약해 주세요", "요약해주세요",
        "지시:", "명령:", "role:", "response:", "output:", "input:",
        "결과:", "요약 결과:", "요약결과:", "정리:", "정리해", "정리해줘",
    )

    lines: List[str] = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue

        low = line.lower()
        if any(low.startswith(p) for p in drop_prefixes):
            continue

        if re.search(r"(다음|아래).{0,10}(요약|정리)", line):
            continue
        if re.search(r"(요약|정리).{0,10}(해줘|해주세요|하세요)", line):
            continue

        lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 2:
        return ""
    return cleaned


def _normalize_whitespace(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        key = x.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


# ============================================
# T5 입력 구성
# ============================================
def _add_t5_task_prefix(corpus: str) -> str:
    corpus = (corpus or "").strip()
    if not corpus:
        return ""
    return f"summarize: {corpus}"


def _build_summary_input(reviews: List[ReviewItem]) -> str:
    parts: List[str] = []
    for r in reviews:
        c = _sanitize_text(r.comment)
        c = _normalize_whitespace(c)
        if c:
            parts.append(c)

    if not parts:
        return ""

    parts = _dedupe_keep_order(parts)
    corpus = "\n\n".join(parts)

    if len(corpus) > 1800:
        corpus = corpus[:1800]

    return _add_t5_task_prefix(corpus)


def _generation_kwargs() -> dict:
    return dict(
        truncation=True,
        max_length=96,
        min_length=28,
        num_beams=6,
        do_sample=False,
        no_repeat_ngram_size=4,
        encoder_no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        length_penalty=1.1,
        early_stopping=True,
    )


# ============================================
# 후처리
# ============================================
def convert_to_polite(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    parts = re.split(r"(\n+|[.!?]+)", s)

    def _polite_sentence(x: str) -> str:
        t = x.strip()
        if not t:
            return ""

        if re.search(r"(습니다|합니다|됩니다|이에요|예요|세요)\b", t):
            return t

        t = re.sub(r"이다$", "입니다", t)
        t = re.sub(r"같다$", "같습니다", t)
        t = re.sub(r"된다$", "됩니다", t)
        t = re.sub(r"한다$", "합니다", t)
        t = re.sub(r"필요하다$", "필요합니다", t)
        t = re.sub(r"좋다$", "좋습니다", t)
        t = re.sub(r"나쁘다$", "나쁩니다", t)
        t = re.sub(r"많다$", "많습니다", t)
        t = re.sub(r"적다$", "적습니다", t)
        t = re.sub(r"해$", "합니다", t)

        if not re.search(r"[.!?]$", t):
            t += "."
        return t

    out: list[str] = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        polished = _polite_sentence(sentence)
        if polished:
            out.append(polished)
        if sep and not polished.endswith((".", "!", "?")):
            out.append(sep)

    result = " ".join(x.strip() for x in out if x.strip())
    result = re.sub(r"\s+", " ", result).strip()
    return result


def fix_korean_spacing(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    return _KIWI.space(s, reset_whitespace=False)


def finalize_punctuation_spacing(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+([.,!?])", r"\1", s)
    s = re.sub(r"([.!?])([^\s])", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================
# Domain Service
# ============================================

_TONE_TO_KR = {
    "POSITIVE": "긍정적",
    "NEUTRAL": "중립",
    "NEGATIVE": "부정적",
}

class ReviewInsightService:
    def __init__(self) -> None:
        self._java_base = settings.JAVA_INTERNAL_BASE_URL
        self._timeout = httpx.Timeout(
            connect=settings.HTTP_CONNECT_TIMEOUT,
            read=settings.HTTP_READ_TIMEOUT,
            write=settings.HTTP_READ_TIMEOUT,
            pool=settings.HTTP_CONNECT_TIMEOUT,
        )

    # ---------- Validation ----------
    def _validate_reviews(self, reviews: list[ReviewItem]) -> None:
        print("DEBUG len(reviews)=", len(reviews),
              "MAX_REVIEWS=", settings.MAX_REVIEWS,
              "MAX_COMMENT_LEN=", settings.MAX_COMMENT_LEN,
              "first_comment_len=", len((reviews[0].comment or "")) if reviews else 0)
        if len(reviews) > settings.MAX_REVIEWS:
            raise HTTPException(status_code=400, detail=f"reviews exceeds limit: {settings.MAX_REVIEWS}")
        for r in reviews:
            c = (r.comment or "").strip()
            if len(c) > settings.MAX_COMMENT_LEN:
                raise HTTPException(status_code=400, detail=f"comment too long (max {settings.MAX_COMMENT_LEN})")

    # ---------- Core 분석 ----------
    def summarize_reviews(self, reviews: list[ReviewItem]) -> str:
        self._validate_reviews(reviews)

        input_text = _build_summary_input(reviews)
        if not input_text:
            return ""

        summarizer = get_summarizer()
        out = summarizer(input_text, **_generation_kwargs())

        summary = (out[0].get("summary_text") or "").strip()
        summary = _normalize_whitespace(summary)
        summary = convert_to_polite(summary)
        summary = _normalize_whitespace(summary)
        summary = fix_korean_spacing(summary)
        summary = finalize_punctuation_spacing(summary)
        return summary

    async def analyze(self, reviews: list[ReviewItem]) -> ReviewAnalyzeResponse:
        # 1) validate
        self._validate_reviews(reviews)

        # 2) mood: 별점 기반 (빠르고 안정적)
        tone = aggregate_tone(reviews)          # POSITIVE/NEUTRAL/NEGATIVE
        mood = _TONE_TO_KR.get(tone, "중립")    # 응답 예시에 맞춘 한글

        # 3) insight: T5 요약
        insight = self.summarize_reviews(reviews)

        return ReviewAnalyzeResponse(mood=mood, insightSummary=insight)

