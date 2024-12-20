# AstuteRAG

AstuteRAG는 LLM(Large Language Model)의 내부 지식과 외부 검색 문서를 효과적으로 통합하여 더 정확하고 신뢰성 있는 답변을 생성하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 특징

- **내부 지식 활용**: LLM의 내부 지식을 활용하여 외부 문서에서 누락된 정보를 보완
- **지식 통합**: 내부 및 외부 문서의 정보를 일관성 있게 통합
- **신뢰도 평가**: 답변 생성 시 신뢰도 점수와 근거를 함께 제공
- **모순 해결**: 문서 간 상충되는 정보를 식별하고 개별적으로 처리

## 설치 방법

```bash
pip install openai
```

## 사용 방법

```python
from astute_rag import AstuteRAG

# AstuteRAG 초기화
rag = AstuteRAG(
    api_key="your-openai-api-key",
    model="gpt-4",  # 또는 "gpt-3.5-turbo"
    max_generated_passages=1
)

# 질문과 검색된 문서로 답변 생성
question = "What is the capital of France?"
retrieved_docs = [
    "Paris is the capital and largest city of France.",
    "The city of Paris serves as France's capital, located in the north.",
]

answer = rag.answer_question(question, retrieved_docs)
print(answer)
```

## 주요 구성 요소

1. **내부 지식 생성 (generate_internal_knowledge)**
   - LLM의 내부 지식을 활용하여 질문과 관련된 정보 생성
   - 불확실한 정보는 "I don't know"로 표시하여 환각 방지

2. **지식 통합 (consolidate_knowledge)**
   - 내부 및 외부 문서의 정보를 군집화
   - 일관된 정보는 통합, 상충되는 정보는 분리

3. **최종 답변 생성 (finalize_answer)**
   - 통합된 지식을 기반으로 최종 답변 생성
   - 답변의 신뢰도 평가 및 근거 제시

## 환경 설정

- Python 3.7 이상
- OpenAI API 키 필요
- 지원 모델: GPT-4, GPT-3.5-turbo

## 참고 문헌

이 프로젝트는 다음 논문을 기반으로 구현되었습니다:
- Wang, Ruochen, et al. "AstuteRAG: Astutely Leveraging Language Model's Internal Knowledge in Retrieval Augmented Generation." arXiv:2410.07176 (2023)

## 라이선스

MIT License

Copyright (c) 2024 AstuteRAG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.