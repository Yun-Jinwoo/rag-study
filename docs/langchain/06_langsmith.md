# LangSmith — 관측성, 평가, 배포

> 공식 문서: https://docs.langchain.com/ (LangSmith 섹션)

---

## 1. LangSmith란?

LangSmith는 LLM 애플리케이션의 **개발·테스트·모니터링·배포를 위한 통합 플랫폼**이다.
RAG 파이프라인 및 에이전트 디버깅에 특히 유용하다.

### HIPAA, SOC 2 Type 2, GDPR 인증 완료

---

## 2. 핵심 기능

### 2-1. Observability (관측성/트레이싱)

에이전트가 어떻게 생각하고 행동하는지 **상세한 실행 추적**을 제공한다.

```bash
# 환경 변수 설정만으로 자동 활성화
export LANGSMITH_API_KEY="ls__your_api_key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="my-rag-project"
```

자동 추적되는 항목:
- LLM 호출 입/출력
- 도구 호출 및 결과
- 각 단계별 소요 시간
- 토큰 사용량
- 에러 및 예외

### 2-2. Evaluation (평가)

에이전트 동작을 **데이터셋에 대해 체계적으로 테스트**하고 점수화한다.

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 평가 데이터셋 생성
dataset = client.create_dataset("rag-qa-dataset")
client.create_examples(
    inputs=[
        {"question": "What is RAG?"},
        {"question": "How does LangChain work?"},
    ],
    outputs=[
        {"answer": "RAG stands for Retrieval-Augmented Generation..."},
        {"answer": "LangChain is a framework for building LLM applications..."},
    ],
    dataset_id=dataset.id,
)

# 평가 함수 정의
def correctness_evaluator(run, example):
    """평가 기준 함수."""
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    # 평가 로직
    score = compute_similarity(prediction, reference)
    return {"key": "correctness", "score": score}

# 실행 및 평가
results = evaluate(
    lambda inputs: agent.invoke(inputs),
    data="rag-qa-dataset",
    evaluators=[correctness_evaluator],
    experiment_prefix="rag-v1",
)
```

### 2-3. Prompt Engineering

- 프롬프트 버전 관리
- A/B 테스트
- 팀 협업

```python
from langsmith import Client

client = Client()

# 프롬프트 저장
client.push_prompt("my-rag-prompt", object=my_prompt_template)

# 프롬프트 불러오기
prompt = client.pull_prompt("my-rag-prompt:v3")
```

### 2-4. Deployment (배포)

LangSmith를 통해 에이전트를 **1클릭으로 배포**하고 확장 가능한 인프라에서 운영.

---

## 3. 평가 방식 유형

### 코드 기반 평가 (Code-based)

```python
def exact_match(run, example):
    return {
        "key": "exact_match",
        "score": int(run.outputs["answer"] == example.outputs["answer"])
    }
```

### LLM-as-Judge 평가

```python
from langsmith.evaluation import LangChainStringEvaluator

correctness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "correctness": "Is the response factually correct based on the reference?"
        }
    }
)
```

### 복합 평가 (Composite)

여러 평가 기준을 결합하여 종합 점수 산출.

---

## 4. 자가 호스팅 배포

```yaml
# Kubernetes
helm repo add langsmith https://langchain-ai.github.io/helm-charts/
helm install langsmith langsmith/langsmith -f values.yaml

# Docker Compose
docker-compose -f langsmith-docker-compose.yml up -d
```

지원 플랫폼: Kubernetes, AWS EKS, Azure AKS, GCP GKE, Docker

---

## 5. RAG 프로젝트에서의 활용

```python
# LangSmith 트레이싱 설정
import os
os.environ["LANGSMITH_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-qna-system"

# 이후 모든 에이전트/체인 실행이 자동으로 트레이싱됨
result = rag_agent.invoke({"messages": [{"role": "user", "content": "What is RAG?"}]})

# LangSmith 대시보드에서 확인:
# - 검색된 문서
# - LLM 프롬프트/응답
# - 전체 실행 시간
# - 각 단계별 토큰 사용량
```

---

## 6. Ragas와의 연동

LangSmith 트레이싱 데이터를 Ragas 평가 메트릭과 연결하여
RAG 시스템의 품질을 지속적으로 모니터링할 수 있다.

```python
# LangSmith에서 실행 데이터 추출
runs = client.list_runs(project_name="rag-qna-system")

# Ragas 메트릭으로 평가
from ragas.metrics.collections import Faithfulness, ContextPrecision

for run in runs:
    score = await Faithfulness(llm=llm).ascore(
        user_input=run.inputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["contexts"]
    )
    print(f"Faithfulness: {score.value:.3f}")
```
