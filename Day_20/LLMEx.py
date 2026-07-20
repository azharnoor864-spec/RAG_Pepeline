# compressed_retriever.py
# ============================================
# STEP 3: Contextual Compression Retriever
# Retrieved chunks ke andar se sirf RELEVANT part nikalta hai,
# baaki irrelevant/extra text hata deta hai - LLM ko bhejne se pehle
# ============================================

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from landchain import build_retriever

load_dotenv()


def build_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def build_compressed_retriever():
    """
    Normal retriever ke upar ek 'compression layer' laga deta hai.
    Flow: query -> normal retriever (5 chunks) -> LLM har chunk padhta
    hai -> sirf relevant sentences wapas aati hain (irrelevant hata di
    jati hain).
    """
    base_retriever = build_retriever()
    llm = build_llm()

    # LLMChainExtractor: har retrieved chunk ko LLM se "compress" karwata
    # hai - poora chunk padh kar sirf directly-relevant lines nikalta hai
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    return compression_retriever


def compare_normal_vs_compressed(query: str):
    """Side-by-side dikhata hai: normal retrieval vs compressed retrieval"""
    base_retriever = build_retriever()
    compressed_retriever = build_compressed_retriever()

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)

    # --- Normal retrieval (bina compression ke) ---
    print("\n--- NORMAL RETRIEVAL (raw chunks, koi filtering nahi) ---\n")
    normal_results = base_retriever.invoke(query)
    for i, doc in enumerate(normal_results, 1):
        print(f"[Chunk {i}] chars={len(doc.page_content)}")
        print(doc.page_content[:300])
        print()

    # --- Compressed retrieval ---
    print("\n--- COMPRESSED RETRIEVAL (sirf relevant part) ---\n")
    compressed_results = compressed_retriever.invoke(query)
    for i, doc in enumerate(compressed_results, 1):
        print(f"[Chunk {i}] chars={len(doc.page_content)}")
        print(doc.page_content)
        print()

    print(f"\nSummary: Normal me {len(normal_results)} chunks the, "
          f"Compressed me {len(compressed_results)} chunks bache "
          f"(kuch chunks poori tarah irrelevant nikal kar hata bhi diye ja sakte hain)")


if __name__ == "__main__":
    # IMPORTANT: LLM ko bheji jane wali query hamesha clean English mein
    # honi chahiye. Mixed-language (Roman Urdu + English) query se LLM
    # confuse hota hai aur extraction/compression format follow karne mein
    # aur zyada masla karta hai.
    test_query = "What was Harry Klopf's contribution to reinforcement learning?"
    compare_normal_vs_compressed(test_query)