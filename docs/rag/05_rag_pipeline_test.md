# RAG 파이프라인 테스트 (BGE-M3 + ChromaDB + LangChain)

> 작성일: 2026-04-05
> 사용 기술: BGE-M3, ChromaDB, LangChain (LCEL)

---

## 1. 개요

BGE-M3 임베딩 모델과 ChromaDB를 LangChain으로 연결하여 의약품 검색 RAG 파이프라인을 구성하고 테스트했다.

**테스트 시나리오:**
> "임산부가 두통이 심할 때, 처방전 없이 살 수 있는 약을 추천해달라."

---

## 2. 전체 파이프라인 구조

```
사용자 질문
    ↓
[BGE-M3]  질문 → 벡터 변환
    ↓
[ChromaDB]  메타데이터 필터링 + 유사 문서 검색
    ↓
검색된 문서 반환
    ↓
[LLM]  문서 기반 답변 생성  ← 추후 구현 예정
```

---

## 3. 환경 설정

```python
!pip install langchain langchain-community langchain-huggingface langchain-chroma
!pip install chromadb sentence-transformers
```

---

## 4. 전체 코드

### 4-1. 임베딩 모델 설정 (BGE-M3)

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # 정규화 → Dot Product = Cosine 유사도
)
```

- `normalize_embeddings=True`: 벡터를 정규화하면 내적(Dot Product)이 코사인 유사도와 동일해져 검색 속도 최적화

---

### 4-2. 의약품 데이터 준비 + ChromaDB 저장

```python
from langchain_core.documents import Document
from langchain_chroma import Chroma

documents = [
    Document(
        page_content="타이레놀(아세트아미노펜)은 해열 및 진통 효과가 있습니다. 두통, 발열에 효과적이며 임산부도 복용 가능합니다. 1회 1~2정, 1일 3회 복용합니다.",
        metadata={"drug_name": "타이레놀", "category": "진통제", "requires_prescription": False, "safe_for_pregnant": True}
    ),
    Document(
        page_content="이부프로펜은 소염진통제로 두통, 치통, 근육통, 생리통에 효과적입니다. 임산부는 복용을 피해야 합니다. 1회 1정, 1일 3회 복용합니다.",
        metadata={"drug_name": "이부프로펜", "category": "소염진통제", "requires_prescription": False, "safe_for_pregnant": False}
    ),
    Document(
        page_content="게보린은 두통, 치통, 생리통에 사용되는 복합 진통제입니다. 임산부 복용 금지. 1회 1정, 1일 3회 복용합니다.",
        metadata={"drug_name": "게보린", "category": "진통제", "requires_prescription": False, "safe_for_pregnant": False}
    ),
    Document(
        page_content="아목시실린은 항생제로 세균 감염 치료에 사용됩니다. 반드시 의사의 처방이 필요합니다.",
        metadata={"drug_name": "아목시실린", "category": "항생제", "requires_prescription": True, "safe_for_pregnant": False}
    ),
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="medicine_rag"
)
```

**메타데이터 구조:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `drug_name` | str | 약품명 |
| `category` | str | 분류 (진통제, 항생제 등) |
| `requires_prescription` | bool | 처방전 필요 여부 |
| `safe_for_pregnant` | bool | 임산부 복용 가능 여부 |

---

### 4-3. Retriever 설정 (메타데이터 필터링 포함)

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2,
        "filter": {
            "$and": [
                {"requires_prescription": {"$eq": False}},
                {"safe_for_pregnant": {"$eq": True}}
            ]
        }
    }
)
```

- ChromaDB에서 조건 여러 개를 필터링할 때는 `$and` 연산자 사용
- `k=2`: 유사한 문서 최대 2개 반환

---

### 4-4. 프롬프트 템플릿

```python
from langchain_core.prompts import PromptTemplate

prompt_template = """
당신은 의약품 정보를 안내하는 전문 상담사입니다.
아래 참고 문서를 바탕으로 질문에 답변해주세요.
참고 문서에 없는 내용은 절대 답변하지 마세요.

참고 문서:
{context}

질문: {question}

아래 형식으로 답변해주세요:
- 약 이름:
- 주요 성분:
- 효능:
- 복용 방법:
- 주의사항:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
```

---

### 4-5. LCEL RAG 체인 구성

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

### 4-6. 검색 실행 및 결과

```python
query = "저 임산부인데 두통이 심해요. 처방전 없이 살 수 있는 약 있나요?"
docs = retriever.invoke(query)

print(f"🔍 질문: {query}\n")
print("📄 검색된 문서:")
for i, doc in enumerate(docs):
    print(f"\n{i+1}위: {doc.page_content}")
    print(f"   메타데이터: {doc.metadata}")
```

**실행 결과:**

```
🔍 질문: 저 임산부인데 두통이 심해요. 처방전 없이 살 수 있는 약 있나요?

📄 검색된 문서:

1위: 타이레놀(아세트아미노펜)은 해열 및 진통 효과가 있습니다. 두통, 발열에 효과적이며 임산부도 복용 가능합니다. 1회 1~2정, 1일 3회 복용합니다.
   메타데이터: {'drug_name': '타이레놀', 'category': '진통제', 'requires_prescription': False, 'safe_for_pregnant': True}

2위: 타이레놀(아세트아미노펜)은 해열 및 진통 효과가 있습니다. 두통, 발열에 효과적이며 임산부도 복용 가능합니다. 1회 1~2정, 1일 3회 복용합니다.
   메타데이터: {'drug_name': '타이레놀', 'category': '진통제', 'requires_prescription': False, 'safe_for_pregnant': True}
```

---

## 5. 결과 분석

### 검색 결과 해석

| 약품 | 검색 결과 | 이유 |
|------|-----------|------|
| 타이레놀 | ✅ 반환됨 | 임산부 안전 + 처방전 불필요 + 두통 관련 |
| 이부프로펜 | ❌ 필터링됨 | `safe_for_pregnant: False` |
| 게보린 | ❌ 필터링됨 | `safe_for_pregnant: False` |
| 아목시실린 | ❌ 필터링됨 | `requires_prescription: True` |

### 1위, 2위가 동일한 이유

현재 데이터셋에 필터 조건(`임산부 안전 + 처방전 불필요`)을 만족하는 문서가 타이레놀 하나뿐이라 `k=2` 요청 시 같은 문서가 중복 반환됨. 실제 데이터가 충분히 쌓이면 해소됨.

---

## 6. LLM을 사용하지 않은 이유

RAG 파이프라인 테스트 과정에서 여러 LLM을 시도했으나 아래 이유로 이번 테스트에서는 제외했다.

| LLM | 시도 결과 | 원인 |
|-----|-----------|------|
| Gemini 2.0 Flash | ❌ 할당량 초과 | Google 계정 billing 활성화로 무료 티어 차단 |
| Gemini 2.0 Flash Lite | ❌ 동일 오류 | 동일 원인 |
| HuggingFace Zephyr-7B | ❌ 모델 오류 | featherless-ai 프로바이더 미지원 |
| google/flan-t5-large | ❌ 한국어 불가 | 한국어 생성 능력 없음, 숫자 나열 출력 |
| TinyLlama-1.1B | ❌ 한국어 불가 | 프롬프트 반복 출력, 생성 품질 부족 |

**결론:** RAG의 핵심은 올바른 문서를 검색하는 것이며, 이번 테스트에서 검색 파이프라인(BGE-M3 + ChromaDB + 메타데이터 필터링)이 정상 작동함을 확인했다. LLM 생성 단계는 추후 계획에서 해결한다.

---

## 7. 추후 계획

### LLM 연동
팀 논의 후 아래 중 하나로 결정하여 LLM 연동 예정.

| 옵션 | 비용 | 비고 |
|------|------|------|
| OpenAI GPT-4o-mini | 유료 (저렴) | 팀 지원금 사용 예정, 멘토님 추천 |
| OpenAI GPT-4o | 유료 | 고품질 필요 시 |
| Gemini | 유료 | billing 활성화 후 사용 가능 |

LLM 교체 시 변경되는 코드는 단 한 줄:

```python
# 현재 (검색만)
docs = retriever.invoke(query)

# 추후 (LLM 연동)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
result = rag_chain.invoke(query)
```

### 데이터 확장
- 현재: 임의 작성 더미 데이터 4개
- 추후: 약학 정보원, 의약품 안전나라 실제 데이터 수집 및 적재
- 메타데이터 항목 확장 검토: 제형, 상비약 여부, 연령대 제한 등

### 파이프라인 고도화 (멘토님 조언 반영)
- 사용자 의도 파악 단계 추가
- 사용자 입력에서 조건(증상, 알러지, 임산부 여부, 연령대) 추출
- Output Format 프롬프트 고정 (약 이름 / 성분 / 효능 / 복용법 / 부작용)
- 대체 약 추천 기능

---

## 8. 참고

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [ChromaDB 공식 문서](https://docs.trychroma.com)
- [LangChain LCEL 문서](https://python.langchain.com/docs/expression_language)
