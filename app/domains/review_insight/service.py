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
    # None이거나 공백이면 바로 "" 반환
    t = (text or "").strip()
    if not t:
        return ""

    # 위험/잡음 패턴 제거 (코드 블록 전체 제거)
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`+", " ", t)

    # 모델 토큰 흔적 제거
    t = t.replace("<s>", " ").replace("</s>", " ")
    t = t.replace("▁", " ")
    t = re.sub(r"[#]{2,}", " ", t)

    # 프롬프트 주입/지시문으로 보이는 라인 제거
    drop_prefixes = (
        "prompt:", "instruction:", "system:", "assistant:", "user:",
        "요약:", "요약해", "요약해줘", "요약해 주세요", "요약해주세요",
        "지시:", "명령:", "role:", "response:", "output:", "input:",
        "결과:", "요약 결과:", "요약결과:", "정리:", "정리해", "정리해줘",
    )

    lines: List[str] = []
    # 줄단위 필터링 (줄 단위로 쪼개서 비어있는 줄 제거)
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue

        # prefix로 시작하면 그 줄은 페기
        low = line.lower()
        if any(low.startswith(p) for p in drop_prefixes):
            continue

        # 한국어 지시문 패턴 제거 (중간에 10글자 낀 것도 제거)
        if re.search(r"(다음|아래).{0,10}(요약|정리)", line):
            continue
        if re.search(r"(요약|정리).{0,10}(해줘|해주세요|하세요)", line):
            continue

        # 필터를 통과한 실제 리뷰 내용만 누적
        lines.append(line)

    # 최종 합치기 + 공백 정리 + 최소 길이 제한
    # 라인들을 공백으로 이어 붙여서 한 문장 덩어리로 만듬
    cleaned = " ".join(lines)
    # 공백이 여러 개면 하나로 축약
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 너무 짧으면 버림
    if len(cleaned) < 2:
        return ""
    return cleaned

# 공백을 한 칸으로 통일
def _normalize_whitespace(s: str) -> str:

    return " ".join((s or "").split()).strip()

# 중복 리뷰 제거(순서 유지)
# - 동일 리뷰 반복 입력으로 인한 요약 편향 줄임
def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        # 공백이면 무시
        key = x.strip()
        if not key:
            continue
        # 같은 내용의 리뷰 코멘트가 여러 번 있으면 첫 번째만 남김
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


# ============================================
# T5 입력 구성
# ============================================

# 이 입력은 "요약" 작업임을 명시
def _add_t5_task_prefix(corpus: str) -> str:
    corpus = (corpus or "").strip()
    if not corpus:
        return ""
    # task prefix(예: summarize: / translate:)
    return f"summarize: {corpus}"


# 모델에 넣어도 되는 리뷰 문장들 쌓음
def _build_summary_input(reviews: List[ReviewItem]) -> str:
    parts: List[str] = []
    for r in reviews:
        # 각 리뷰의 comment를 가져와서 잡음 제거
        c = _sanitize_text(r.comment)
        # 공백 정규화
        c = _normalize_whitespace(c)
        if c:
            parts.append(c)

    if not parts:
        return ""

    parts = _dedupe_keep_order(parts)

    # 요약 모델(T5)에 넣을 최종 입력 텍스트
    # - 리뷰들을 공백으로 붙이지 않고 빈 줄로 구문해서 합침
    # - 모델 입장에서 "문장 덩어리"로 인식하기 쉬워 요약 품질 향상
    corpus = "\n\n".join(parts)

    # 길이 제한
    if len(corpus) > 1800:
        corpus = corpus[:1800]

    # 최종: T5 prefix 붙여서 반환
    return _add_t5_task_prefix(corpus)


def _generation_kwargs() -> dict:
    return dict(
        truncation=True, # 입력이 모델 한도를 넘으면 잘라서라도 실행
        max_length=96, # 너무 짧은 요약 방지
        min_length=28, # 너무 긴 요약 방지
        num_beams=6, # 같은 입력이면 결과가 거의 동일하게
        do_sample=False, # 랜덤성 없음
        no_repeat_ngram_size=4, # 출력에서 반복 줄이기 "~합니다 ~합니다 ~합니다"
        encoder_no_repeat_ngram_size=4, # 입력에 있던 동일 구절을 출력해서 과하게 베끼는 것 억제
        repetition_penalty=1.15, # 반복 생성에 패널티
        length_penalty=1.1, # 너무 짧게 끝나는 것 방지
        early_stopping=True, # 조건 만족 시 조기 종료
    )


# ============================================
# 후처리
# ============================================

# 존댓말로 종결어미 정리
def convert_to_polite(text: str) -> str:
    # 입력이 비면 종료
    s = (text or "").strip()
    if not s:
        return ""

    # 문장 단위로 쪼개기
    parts = re.split(r"(\n+|[.!?]+)", s)

    # 문장 하나를 존댓말로 변환
    def _polite_sentence(x: str) -> str:
        t = x.strip()
        if not t:
            return ""

        # 이미 존댓말이면 변환하지 않음
        if re.search(r"(습니다|합니다|됩니다|이에요|예요|세요)\b", t):
            return t

        # 문장 끝($)에만 매칭해서 변환
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

        # 문장 부호가 없으면 마침표를 붙여서 문장 형태로 정리
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

        # 문장부호 중복 방지
        if sep and not polished.endswith((".", "!", "?")):
            out.append(sep)

    # 줄바꿈/여러 공백을 정돈
    result = " ".join(x.strip() for x in out if x.strip())
    result = re.sub(r"\s+", " ", result).strip()
    return result


# Kiwi로 띄어쓰기 보정
def fix_korean_spacing(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    return _KIWI.space(s, reset_whitespace=False) # 이미 존재하는 공백에 대해 스스로 판단해서 보정하게 함


# 문장 부호 주변 공백
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

# API 최종 응답용 매핑 테이블
_TONE_TO_KR = {
    "POSITIVE": "긍정적",
    "NEUTRAL": "중립",
    "NEGATIVE": "부정적",
}

class ReviewInsightService:
    def __init__(self) -> None:
        #self._java_base = settings.JAVA_INTERNAL_BASE_URL # Java로 콜백 준비 (현재 사용 되지 않음)
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
        # 리뷰 개수 제한
        if len(reviews) > settings.MAX_REVIEWS:
            raise HTTPException(status_code=400, detail=f"reviews exceeds limit: {settings.MAX_REVIEWS}")
        # 코멘트 길이 제한
        for r in reviews:
            c = (r.comment or "").strip()
            if len(c) > settings.MAX_COMMENT_LEN:
                raise HTTPException(status_code=400, detail=f"comment too long (max {settings.MAX_COMMENT_LEN})")

    # ---------- Core 분석 ----------
    # insightSummary (요약문) 생성
    def summarize_reviews(self, reviews: list[ReviewItem]) -> str:
        self._validate_reviews(reviews)

        # 리뷰 코멘트를 전처리/중복제거/합치기 후 T5 입력 문자열 생성.
        input_text = _build_summary_input(reviews)
        if not input_text:
            return ""

        summarizer = get_summarizer() # 프로세스에서 1번만 pipeline을 생성해 재사용.
        out = summarizer(input_text, **_generation_kwargs()) # 결과 안정화

        # 모델 출력 꺼냄
        summary = (out[0].get("summary_text") or "").strip()

        # 후처리 체인 (읽기 좋게 다듬기)
        summary = _normalize_whitespace(summary)
        summary = convert_to_polite(summary)
        summary = _normalize_whitespace(summary)
        summary = fix_korean_spacing(summary)
        summary = finalize_punctuation_spacing(summary)
        return summary

    # 최종 응답 조합
    async def analyze(self, reviews: list[ReviewItem]) -> ReviewAnalyzeResponse:
        # 1) validate
        self._validate_reviews(reviews)

        # 2) mood: 별점 기반 (빠르고 안정적)
        tone = aggregate_tone(reviews)  # 리뷰들의 rating 평균으로 tone을 결정(텍스트 미사용).
        mood = _TONE_TO_KR.get(tone, "중립") # 그 tone을 _TONE_TO_KR로 한글로 바꿔 mood에 넣음.

        # 3) insight: T5 요약
        insight = self.summarize_reviews(reviews)

        # 최종 응답
        return ReviewAnalyzeResponse(mood=mood, insightSummary=insight)

