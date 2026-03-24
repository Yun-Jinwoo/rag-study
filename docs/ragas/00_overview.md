# Ragas 개요 (Overview)

> **참고 문서**: https://docs.ragas.io/en/stable/

---

## Ragas란?

**Ragas**는 LLM(대규모 언어 모델) 기반 애플리케이션을 체계적으로 평가하기 위한 오픈소스 라이브러리다.
단순한 "느낌(vibe check)"에 의존하는 주관적인 평가를 넘어, **반복 가능하고 정량적인 평가 프로세스**를 구현하는 데 초점을 맞추고 있다.

### 핵심 철학

> "측정할 수 없으면 개선할 수 없다."

Ragas는 AI 애플리케이션의 품질을 지속적으로 향상시키기 위한 **평가 → 실험 → 개선의 루프**를 구축하도록 설계되었다.

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **실험 우선 방법론** | 변경 사항을 추적하고, 개선 효과를 측정하며, 버전 간 결과를 비교 |
| **커스터마이징 가능한 메트릭** | 데코레이터 방식 또는 사전 구축된 메트릭 라이브러리 활용 |
| **프레임워크 통합** | LangChain, LlamaIndex, Haystack 등 주요 프레임워크와 연동 |
| **데이터셋 관리** | 평가용 데이터셋 생성, 저장, 버전 관리 |
| **관측 가능성** | LangSmith, Arize Phoenix 등 모니터링 플랫폼 연동 |

---

## Ragas가 다루는 평가 대상

```
LLM 애플리케이션
├── RAG (Retrieval-Augmented Generation) 시스템
├── Agent / 도구 사용 시스템
├── 프롬프트 평가 및 최적화
├── 멀티턴 대화 시스템
└── 텍스트-to-SQL, 요약, 일반 QnA
```

---

## 평가 워크플로우

```
1. 데이터셋 준비/로드
      ↓
2. LLM & 임베딩 모델 설정
      ↓
3. 평가 메트릭 선택
      ↓
4. evaluate() 또는 aevaluate() 실행
      ↓
5. 결과 분석
      ↓
6. (선택) 관측 플랫폼에 업로드
```

---

## 빠른 시작 (Quick Start)

### 설치

```bash
pip install ragas
```

### 기본 평가 예시

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 평가 데이터셋 준비
dataset = [
    {
        "user_input": "What is RAG?",
        "retrieved_contexts": ["RAG stands for Retrieval-Augmented Generation..."],
        "response": "RAG is a technique that combines retrieval and generation.",
        "reference": "RAG (Retrieval-Augmented Generation) is a method..."
    }
]

# 평가 실행
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
```

### CLI 도구

```bash
ragas quickstart    # 프로젝트 템플릿 생성
ragas evals         # 데이터셋에 대한 평가 실행
ragas hello_world   # 검증 예시 실행
```

**CLI 템플릿 목록:**
- `rag_eval` - RAG 시스템 평가
- `improve_rag` - RAG 개선
- `agent_evals` - 에이전트 평가
- `text2sql` - 텍스트-to-SQL 평가
- `prompt_evals` - 프롬프트 평가
- `benchmark_llm` - LLM 벤치마킹
- `judge_alignment` - LLM 판사 정렬

---

## 커뮤니티 & 지원

- **Discord**: 커뮤니티 Q&A
- **GitHub**: 이슈 리포트 및 기능 요청
- **Office Hours**: 오피스 아워 이용 가능
- **문의**: founders@vibrantlabs.com
