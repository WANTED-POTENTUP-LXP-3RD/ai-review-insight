# ai-review-insight

LXP 메인 서버에서 전달받은 리뷰 데이터를 기반으로  
**리뷰 요약 및 인사이트 분석을 수행하는 Python AI 서버**입니다.

> ⚠️ 본 서버는 **DB에 직접 접근하지 않습니다.**  
> 모든 입력/출력은 **메인 서버(API 서버)** 를 통해 이루어집니다.

---

## 📌 역할과 책임

- 리뷰 원문을 입력으로 받아 **요약 문장 및 분위기(Tone) 분석**
- AI/텍스트 분석 전용 서버로서 **비즈니스 로직 및 DB 접근 없음**
- 메인 서버가 제공한 데이터만 처리하고, 결과를 다시 메인 서버로 반환

---

## 🧩 기술 스택

- **Language**: Python 3.11
- **API Framework**: FastAPI, Uvicorn
- **AI Model**
    - `eenzeenee/t5-base-korean-summarization` (리뷰 요약)
- **Text Processing**
    - `kiwipiepy`
    - `regex`
- **Config**
    - `python-dotenv`

---

## ⚙️ 실행 환경 준비

### 1️⃣ Python 3.11 설치 확인
```bash
python3.11 -V
```
### 2️⃣ 가상환경 생성 및 활성화

본 프로젝트는 **로컬 가상환경(venv)** 사용을 전제로 합니다.  
전역 Python 환경에 라이브러리를 설치하지 마세요.

```bash
python3.11 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# Windows
# .venv\Scripts\activate
```
가상환경 활성화 확인:
```bash
python -V
# Python 3.11.x 
# 터미널 프롬프트 앞에 (.venv)가 표시되면 정상
```
### 3️⃣의존성 설치
```bash
pip install -r requirements.txt
```

### 서버 실행
아래 명령은 프로젝트 루트 디렉터리에서 실행합니다.
```bash
uvicorn app.main:app --reload --port 8001
```
정상 실행 로그
```bash
Uvicorn running on http://127.0.0.1:8001
Application startup complete
```
접속 확인
- Swagger UI: http://127.0.0.1:8001/docs

---

### 🔗 메인 서버 연동 
메인 서버(Spring)는 Python AI 서버를 외부 API로 호출합니다.
```bash
AI_SERVER_BASE_URL=http://localhost:8001
```

---

### 🗂️ 프로젝트 구조
```bash
ai-review-insight/
├─ app/
│  ├─ api/
│  │  └─ api.py
│  │     └─ FastAPI 앱 생성(create_api) + 도메인 router들을 include
│  │
│  ├─ core/
│  │  ├─ config.py
│  │  │  └─ 환경변수/.env 로딩, 설정값 관리(HF_MODEL, PORT 등)
│  │  ├─ exceptions.py
│  │  │  └─ 공통 예외 정의 + FastAPI 예외 핸들러(표준 에러 응답)
│  │  └─ logging.py
│  │     └─ 로깅 포맷/레벨/핸들러 설정
│  │
│  ├─ domains/
│  │  └─ review_insight/
│  │     ├─ model.py
│  │     │  └─ (선택) Enum/상수/도메인 규칙(예: Tone) 같은 “순수 모델”
│  │     ├─ schema.py
│  │     │  └─ Request/Response DTO (Pydantic)
│  │     ├─ service.py
│  │     │  └─ 핵심 로직: 전처리 → HF pipeline → 후처리(kiwi/regex)
│  │     ├─ router.py
│  │     │  └─ HTTP 엔드포인트 정의(POST /review-insights/summarize)
│  │     └─ __init__.py
│  │
│  └─ main.py
│     └─ uvicorn 진입점: app = create_api()
│
├─ scripts/
│  └─ smoke_test.py
│     └─ (선택) 로컬 스모크 테스트(kiwi/모델 로딩 확인)
│
├─ .env.template
│  └─ 팀원이 복사해서 .env로 쓰는 템플릿(민감정보 없음)
├─ .gitignore
│  └─ venv/.env/캐시/IDE 파일 제외 규칙
├─ README.md
│  └─ 실행 방법, 환경변수, 테스트 방법
└─ requirements.txt
   └─ 파이썬 의존성 목록
```

---

### 모델/라이브러리 동작 확인
- app/main.py나 FastAPI startup 이벤트에서 이 스크립트를 import/실행하면 안 됩니다.
- 첫 실행은 모델 다운로드로 오래 걸릴 수 있습니다.
```bash
python scripts/smoke_test.py
```