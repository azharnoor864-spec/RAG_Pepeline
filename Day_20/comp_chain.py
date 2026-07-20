# llm_chain_filter_compression.py
# ============================================
# 3RD ALTERNATIVE: LLMChainFilter
# LLMChainExtractor jaisa NAHI - ye chunk ko REWRITE/GENERATE nahi karta.
# Bas LLM se ek simple Yes/No decision leta hai: "relevant hai ya nahi?"
# Agar Yes -> poora chunk as-is rakho. Agar No -> hata do.
# Isliye hallucination ka risk khatam - koi naya text banta hi nahi.
# ============================================

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter

from landchain import build_retriever

load_dotenv()


def build_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def build_llm_filter_retriever():
    base_retriever = build_retriever()
    llm = build_llm()

    # LLMChainFilter: har chunk ke liye LLM se poochta hai "keep or discard?"
    # Koi rewriting nahi hoti - chunk pura ka pura as-is rehta hai agar
    # "keep" decide ho, warna poora hata diya jata hai.
    doc_filter = LLMChainFilter.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=doc_filter,
        base_retriever=base_retriever,
    )

    return compression_retriever


if __name__ == "__main__":
    retriever = build_llm_filter_retriever()

    test_query = "What was Harry Klopf's contribution to reinforcement learning?"
    print(f"Query: {test_query}\n")

    results = retriever.invoke(test_query)
    print(f"Total chunks jo LLM ne 'relevant' declare kiye: {len(results)}\n")

    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source_filename", "N/A")
        page = doc.metadata.get("page_number", "N/A")
        print(f"[Chunk {i}] {src} | page {page} | chars={len(doc.page_content)}")
        print(doc.page_content[:250])
        print()