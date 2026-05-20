# trustOpsBack vLLM Gateway

FastAPI 기반의 OpenAI 호환 LLM 게이트웨이입니다. `vllm-qwen3-5` 서비스로 요청을 전달하고, 선택적으로 Langfuse에 trace를 남깁니다.

## 구조

- `app/main.py`: FastAPI 진입점
- `app/app_factory.py`: 앱 조립과 lifespan 관리
- `app/proxy.py`: vLLM upstream 프록시
- `app/langfuse_recorder.py`: Langfuse trace 기록
- `app/settings.py`: 환경 설정 로딩
- `app/routes.py`: API 라우트 정의

## 환경 변수

`.env.example`을 복사해서 `.env`를 만든 뒤 설정하세요.

- `VLLM_BASE_URL`: vLLM 서버 주소
- `GATEWAY_API_KEY`: 선택적 API 키
- `PROMPT_CONFIG_PATH`: 프롬프트 설정 파일 경로. Kubernetes에서는 `/app/prompt-config.yaml`을 사용하고, 로컬 개발에서는 `./prompt-config.example.yaml`을 사용합니다.
- `LANGFUSE_ENABLED`: Langfuse 활성화 여부
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`: Langfuse 연결 정보

## 실행

### Conda

```bash
conda env create -f environment.yml
conda activate trustopsback
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

로컬 개발에서는 `.env.example`의 `PROMPT_CONFIG_PATH=./prompt-config.example.yaml` 값을 그대로 쓰면 됩니다. 운영 Kubernetes Pod에서는 ConfigMap이 `/app/prompt-config.yaml`에 mount되므로 별도 파일을 만들 필요가 없습니다.

### Pip

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## 엔드포인트

- `GET /health`
- `GET /`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET|POST|PUT|PATCH|DELETE|OPTIONS /v1/{path}`
- `GET|POST|PUT|PATCH|DELETE|OPTIONS /openai/{path}`

## 요청 예시

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-gateway-api-key: <your-key>' \
  -d '{
    "model": "cyankiwi/Qwen3.5-4B-AWQ-4bit",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
