from __future__ import annotations

import os

from kiwipiepy import Kiwi
from transformers import pipeline


def main() -> None:
    # 1) 형태소/띄어쓰기 라이브러리 로딩 확인
    kiwi = Kiwi()
    print("[OK] Kiwi loaded")

    # 2) 모델명은 .env 또는 환경변수로 바꿀 수 있음
    model_name = os.getenv("HF_MODEL", "eenzeenee/t5-base-korean-summarization")
    print(f"[INFO] HF_MODEL={model_name}")

    # 3) HuggingFace 모델 로딩 + pipeline 생성 확인
    summarizer = pipeline("summarization", model=model_name)
    print("[OK] HF pipeline created")

    # 4) 아주 짧은 샘플로 1회 추론(진짜로 동작하는지)
    text = "강의 내용은 좋지만 로딩이 느려서 학습 흐름이 자주 끊깁니다."
    out = summarizer(f"summarize: {text}", max_length=64, min_length=16, do_sample=False)
    print("[RESULT]", out[0]["summary_text"])


if __name__ == "__main__":
    main()
