# day19_rerank.py
# Step 5: Cross-Encoder Re-ranking
# Hybrid Search (BM25 + Vector + RRF) se aaye top-20 candidates ko
# cross-encoder/ms-marco-MiniLM-L-12-v2 se dobara, accurately score karna

import os
import json
import chromadb
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ---- Hardcoded paths ----
CHUNKS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"
QUESTIONS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day_19\testquestion.json"
CHROMA_DB_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"

CHROMA_COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

TOP_N_PER_METHOD = 20   # BM25/Vector se kitne candidates lene hain
RRF_K = 60
HYBRID_TOP_K = 20       # RRF ke baad kitne candidates re-ranker ko dene hain
FINAL_TOP_K = 5         # re-ranking ke baad kitne final results chahiye

GROQ_MODEL = "llama-3.3-70b-versatile"
groq_client = Groq()

REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for a retrieval system.
Your job is to rewrite the user's question into a clearer, more specific version
that will work better for document search (BM25 keyword search and vector/semantic search).

Rules:
- Keep the rewritten query factual and neutral, do not answer the question.
- Expand vague pronouns or references into specific terms if context suggests them.
- Keep important keywords (names, numbers, dates, technical terms) intact.
- Do not add information that isn't implied by the original question.
- Output ONLY the rewritten query, nothing else (no preamble, no quotes).
"""


# ============ QUERY REWRITING ============

def rewrite_query(original_query):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": original_query},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ============ BM25 ============

def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def search_bm25(bm25, chunks, query, top_k):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]


# ============ VECTOR (CHROMADB) ============

def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def search_vector(collection, embed_model, query, top_k):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    return list(zip(documents, distances, metadatas))


# ============ RRF FUSION ============

def get_chunk_key(text):
    return text.strip()


def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    rrf_scores = {}

    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        key = get_chunk_key(chunk["text"])
        if key not in rrf_scores:
            rrf_scores[key] = {"rrf_score": 0.0, "text": chunk["text"],
                                "source": chunk.get("source_filename"), "found_in": []}
        rrf_scores[key]["rrf_score"] += 1 / (k + rank)
        rrf_scores[key]["found_in"].append(f"BM25 (rank {rank})")

    for rank, (text, distance, metadata) in enumerate(vector_results, start=1):
        key = get_chunk_key(text)
        if key not in rrf_scores:
            rrf_scores[key] = {"rrf_score": 0.0, "text": text,
                                "source": metadata.get("source_filename"), "found_in": []}
        rrf_scores[key]["rrf_score"] += 1 / (k + rank)
        rrf_scores[key]["found_in"].append(f"Vector (rank {rank})")

    return rrf_scores


def hybrid_search(bm25, chunks, collection, embed_model, query, top_n=20, final_k=20):
    bm25_results = search_bm25(bm25, chunks, query, top_k=top_n)
    vector_results = search_vector(collection, embed_model, query, top_k=top_n)
    rrf_scores = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:final_k]


# ============ CROSS-ENCODER RE-RANKING (naya, is step ka core) ============

def rerank_with_cross_encoder(cross_encoder, query, candidates, top_k=5):
    """
    candidates: hybrid_search se aayi list [{"text": ..., "source": ..., "rrf_score": ...}, ...]

    Cross-encoder ko (query, chunk_text) pairs diye jate hain, aur woh
    har pair ke liye ek relevance score deta hai. Phir hum us score
    ke hisaab se dobara sort karte hain.
    """
    # Cross-encoder ko pairs chahiye: [[query, text1], [query, text2], ...]
    pairs = [[query, c["text"]] for c in candidates]

    ce_scores = cross_encoder.predict(pairs)

    # Har candidate mein cross-encoder score add karna
    for candidate, ce_score in zip(candidates, ce_scores):
        candidate["cross_encoder_score"] = float(ce_score)

    # Cross-encoder score ke hisaab se dobara sort karna (highest = most relevant)
    reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

    return reranked[:top_k]


# ============ MAIN: REWRITE -> HYBRID -> RE-RANK ============

def main():
    print("Chunks aur questions load ho rahe hain...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"{len(chunks)} chunks aur {len(questions)} questions load hue.\n")

    print("BM25 index ban raha hai...")
    bm25 = build_bm25_index(chunks)

    print(f"Embedding model '{EMBEDDING_MODEL_NAME}' load ho raha hai...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Cross-encoder model '{CROSS_ENCODER_MODEL}' load ho raha hai...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print("ChromaDB collection load ho raha hai...\n")
    collection = load_chroma_collection()

    all_results = []

    for q in questions:
        original_query = q["question"]
        rewritten_query = rewrite_query(original_query)

        print("=" * 80)
        print(f"Q{q['id']} Original : {original_query}")
        print(f"     Rewritten: {rewritten_query}")
        print("=" * 80)

        # ---- STEP 1: Hybrid search se top-20 candidates lena ----
        hybrid_candidates = hybrid_search(
            bm25, chunks, collection, embed_model, rewritten_query,
            top_n=TOP_N_PER_METHOD, final_k=HYBRID_TOP_K
        )

        # ---- STEP 2: Un 20 candidates ko cross-encoder se re-rank karna ----
        reranked = rerank_with_cross_encoder(
            cross_encoder, rewritten_query, hybrid_candidates, top_k=FINAL_TOP_K
        )

        top = reranked[0]
        print(f"\n[Final Top Result - After Re-ranking]")
        print(f"  Cross-Encoder Score: {top['cross_encoder_score']:.4f}")
        print(f"  RRF Score (before rerank): {top['rrf_score']:.5f}")
        print(f"  Source: {top['source']}")
        print(f"  Text: {top['text'][:150]}...")
        print()

        all_results.append({
            "question_id": q["id"],
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "final_top_source": top["source"],
            "final_top_text": top["text"][:200],
            "cross_encoder_score": top["cross_encoder_score"],
            "rrf_score": top["rrf_score"],
        })

    with open("final_reranked_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nDone! Results 'final_reranked_results.json' mein save ho gaye hain.")


if __name__ == "__main__":
    main()