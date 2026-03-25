# RAG (Retrieval-Augmented Generation) 란?

> 개인 학습 정리 노트

---

## 1. 왜 RAG가 필요한가?

LLM(대형 언어 모델)은 강력하지만, 두 가지 근본적인 한계가 있다.

| 한계 | 설명 |
|------|------|
| **지식 단절 (Knowledge Cutoff)** | 학습 데이터 이후의 정보를 알지 못함 |
| **환각 (Hallucination)** | 모르는 내용에 대해 그럴듯하게 거짓 답변을 생성함 |

RAG는 이 두 문제를 **외부 지식을 검색해서 LLM에 주입**하는 방식으로 해결한다.

---

## 2. RAG의 개념

**RAG = Retrieval + Augmented + Generation**

```
Retrieval   : 질문과 관련된 문서를 외부 데이터베이스에서 검색
Augmented   : 검색된 문서를 프롬프트에 추가(보강)
Generation  : LLM이 보강된 프롬프트를 바탕으로 최종 답변 생성
```

### 기본 흐름

```
사용자 질문 (Query)
       ↓
[1. Retrieval]  질문을 벡터로 변환 → Vector DB에서 유사 문서 검색
       ↓
[2. Augment]    검색된 문서 + 질문 → 프롬프트 조합
       ↓
[3. Generation] LLM이 프롬프트를 읽고 답변 생성
       ↓
최종 답변 (Answer)
```

---

## 3. 핵심 아이디어: Embedding과 Vector 유사도 검색

RAG의 검색 단계는 **임베딩(Embedding)** 기술에 기반한다.

### Embedding이란?
- 텍스트(문장, 문서)를 **숫자 벡터**로 변환하는 기술
- 의미가 비슷한 문장은 벡터 공간에서 **가까운 위치**에 놓임
- 예: `"강아지"` 와 `"개"` 는 벡터가 매우 유사함

```python
# 예시: 텍스트 → 벡터
"LangChain은 LLM 프레임워크이다" → [0.12, -0.34, 0.87, ...]  # 수백~수천 차원
```

### 유사도 검색
- 사용자 질문도 벡터로 변환 후, 미리 저장된 문서 벡터들과 유사도 비교
- 가장 유사한 문서 top-K개 반환

```
질문 벡터: [0.10, -0.30, 0.85, ...]
문서 벡터: [0.12, -0.34, 0.87, ...]  ← 유사도 높음 → 검색됨
문서 벡터: [0.90, 0.50, -0.20, ...]  ← 유사도 낮음 → 검색 안됨
```

주요 유사도 측정 방법:
- **Cosine Similarity** (코사인 유사도) — 가장 많이 사용
- **Dot Product** (내적)
- **Euclidean Distance** (L2 거리)

---

## 4. RAG의 세대 분류

### Naive RAG (기본 RAG)
가장 단순한 형태. 검색 → 프롬프트 구성 → 생성 의 단순 파이프라인.

```
Query → Retrieve → Prompt → Generate
```

**단점:** 검색 품질이 낮으면 답변 품질도 낮음, 노이즈에 취약

---

### Advanced RAG (고급 RAG)
검색 전/후 단계를 강화한 형태.

```
Query → [Pre-Retrieval 강화] → Retrieve → [Post-Retrieval 강화] → Generate
```

**Pre-Retrieval 기법:**
- **Query Rewriting** : 질문을 더 검색하기 좋은 형태로 재작성
- **HyDE (Hypothetical Document Embedding)** : 가상의 답변을 먼저 생성하고, 그 답변으로 검색
- **Multi-Query** : 질문을 여러 버전으로 변환해 다각도로 검색

**Post-Retrieval 기법:**
- **Re-ranking** : 검색된 문서를 관련도 순으로 재정렬
- **Compression** : 검색된 문서에서 핵심 내용만 추출

---

### Modular RAG (모듈형 RAG)
각 컴포넌트를 독립적으로 설계하고 조합하는 유연한 구조.
Self-RAG, Corrective RAG 등이 여기에 속함.

**Self-RAG:** LLM이 검색이 필요한지 스스로 판단하고, 생성한 답변이 문서와 일치하는지도 스스로 검증

---

## 5. RAG vs Fine-tuning

| 항목 | RAG | Fine-tuning |
|------|-----|-------------|
| 지식 업데이트 | 실시간 가능 | 재학습 필요 |
| 비용 | 상대적으로 저렴 | 높은 GPU 비용 |
| 투명성 | 출처 추적 가능 | 블랙박스 |
| 전문 도메인 적용 | 쉬움 | 학습 데이터 필요 |
| 환각 억제 | 효과적 | 제한적 |

> 실제로는 RAG + Fine-tuning을 함께 사용하는 경우도 많다.

---

## 6. RAG가 활용되는 분야

- **기업 내부 문서 QnA** (사규, 매뉴얼, 보고서 등)
- **고객 지원 챗봇**
- **법률/의료 도메인 QnA**
- **코드 검색 및 문서 보조**
- **학술 논문 검색 및 요약**

---

## 참고 자료

- [Ragas 공식 문서](https://docs.ragas.io/en/stable/)
- [LangChain RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/)
- [Lewis et al. (2020) - RAG 원논문](https://arxiv.org/abs/2005.11401) [(논문 정리본)](https://github.com/Yun-Jinwoo/rag-study/blob/main/docs/rag/02_rag_original_paper.md)
