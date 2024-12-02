from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from openai import OpenAI
import os

@dataclass
class Document:
    content: str
    source: str  # 'internal' or 'external'
    doc_id: str

class AstuteRAG:
    def __init__(self, api_key: str, model: str = "gpt-4", max_generated_passages: int = 1):
        """
        Initialize AstuteRAG
        Args:
            api_key: OpenAI API key
            model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            max_generated_passages: Maximum number of passages to generate from internal knowledge
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_generated_passages = max_generated_passages

    def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI API with the given prompt
        Args:
            prompt: Input prompt
        Returns:
            LLM response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def generate_internal_knowledge(self, question: str) -> List[Document]:
        """
        Generate passages from LLM's internal knowledge
        """
        prompt = f"""Generate a document that provides accurate and relevant information to answer the given
        question. If the information is unclear or uncertain, explicitly state 'I don't know' to avoid any
        hallucinations.
        
        Question: {question}
        Document:"""
        
        response = self._call_llm(prompt)
        
        if "I don't know" not in response:
            doc = Document(
                content=response,
                source="internal",
                doc_id=f"internal_0"
            )
            return [doc]
        return []

    def consolidate_knowledge(self, 
                            question: str,
                            documents: List[Document]) -> List[Document]:
        """
        Consolidate information from both internal and external documents
        """
        context = "\n\n".join([f"Document {i} ({doc.source}): {doc.content}" 
                             for i, doc in enumerate(documents)])
        
        prompt = f"""Task: Consolidate information from both memorized documents and externally
        retrieved documents in response to the given question.
        
        For documents that provide consistent information, cluster them together.
        For documents with conflicting information, separate them into distinct documents.
        Exclude any irrelevant information.
        
        Question: {question}
        Context: {context}
        
        Provide consolidated documents in JSON format:
        [{{"content": "consolidated content", "source": ["doc_ids"], "consistency_group": "group_id"}}]"""
        
        response = self._call_llm(prompt)
        
        try:
            consolidated = json.loads(response)
            return [Document(
                content=doc["content"],
                source=",".join(doc["source"]),
                doc_id=f"consolidated_{doc['consistency_group']}"
            ) for doc in consolidated]
        except:
            return documents

    def finalize_answer(self, 
                       question: str,
                       initial_docs: List[Document],
                       consolidated_docs: List[Document]) -> str:
        """
        Generate final answer based on consolidated knowledge
        """
        initial_context = "\n\n".join([f"Initial Document {i} ({doc.source}): {doc.content}"
                                     for i, doc in enumerate(initial_docs)])
        consolidated_context = "\n\n".join([f"Consolidated Document {i} ({doc.source}): {doc.content}"
                                          for i, doc in enumerate(consolidated_docs)])
        
        prompt = f"""Task: Answer the question using consolidated information from both internal
        and external documents.
        
        Initial Context: {initial_context}
        
        Consolidated Context: {consolidated_context}
        
        Question: {question}
        
        Provide your answer in the format:
        {{
            "answer": "final answer",
            "confidence": "confidence score and reasoning"
        }}"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            return result["answer"]
        except:
            return response

    def answer_question(self, question: str, retrieved_docs: List[str]) -> str:
        """
        Main method to answer a question using Astute RAG
        """
        # Convert retrieved docs to Document objects
        external_docs = [
            Document(content=doc, source="external", doc_id=f"external_{i}")
            for i, doc in enumerate(retrieved_docs)
        ]
        
        # Step 1: Generate internal knowledge
        internal_docs = self.generate_internal_knowledge(question)
        
        # Step 2: Combine internal and external documents
        all_docs = external_docs + internal_docs
        
        # Step 3: Consolidate knowledge
        consolidated_docs = self.consolidate_knowledge(question, all_docs)
        
        # Step 4: Generate final answer
        answer = self.finalize_answer(question, all_docs, consolidated_docs)
        
        return answer

def main():
    # API 키 설정
    api_key = "your-openai-api"  # 여기에 실제 API 키를 입력하세요
    
    # AstuteRAG 초기화
    rag = AstuteRAG(
        api_key=api_key,
        model="gpt-4",  # 또는 "gpt-3.5-turbo"
        max_generated_passages=1
    )
    
    # 예제 질문과 검색된 문서들
    question = "What is the capital of France?"
    retrieved_docs = [
        "Paris is the capital and largest city of France.",
        "The city of Paris serves as France's capital, located in the north.",
        "France's political center is Paris, which became the capital in 987."
    ]
    
    # 테스트할 다양한 질문과 컨텍스트 준비
    test_cases = [
        {
            "question": "What is quantum computing?",
            "retrieved_docs": [
                "Quantum computing uses quantum phenomena like superposition and entanglement to perform calculations.",
                "Quantum computers can solve certain problems exponentially faster than classical computers.",
                "IBM and Google are leading companies in quantum computer development."
            ]
        },
        {
            "question": "When was the COVID-19 vaccine first administered?",
            "retrieved_docs": [
                "The first COVID-19 vaccine was administered in the UK on December 8, 2020, to Margaret Keenan.",
                "Mass vaccination programs began in December 2020 across multiple countries.",
                "Pfizer-BioNTech's vaccine was the first to receive emergency authorization."
            ]
        }
    ]
    
    print("\n=== RAG Performance Analysis ===\n")

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        retrieved_docs = test_case["retrieved_docs"]
        
        print(f"\nTest Case {i}:")
        print(f"Question: {question}")
        
        print("\n1. Without RAG (Using only LLM):")
        try:
            # RAG 없이 내부 지식만으로 답변
            internal_answer = rag._call_llm(f"Answer this question: {question}")
            print(f"Answer: {internal_answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
            
        print("\n2. With RAG (Using both internal knowledge and external documents):")
        try:
            # RAG를 사용하여 답변
            rag_answer = rag.answer_question(question, retrieved_docs)
            print(f"Answer: {rag_answer}")
        except Exception as e:
            print(f"Error: {str(e)}")
    

if __name__ == "__main__":
    main()