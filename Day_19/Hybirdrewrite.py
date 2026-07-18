# day19_hybrid_with_rewrite.py
# Step 4b: Query Rewriting + Hybrid Search (BM25 + Vector + RRF)
# Ab query pehle LLM se rewrite hogi, phir hybrid retrieval chalegi

import os
import json
import chromadb
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---- Hardcoded paths ----
CHUNKS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"
QUESTIONS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day_19\testquestion.json"
CHROMA_DB_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"

CHROMA_COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

TOP_N_PER_METHOD = 20
RRF_K = 60
FINAL_TOP_K = 5

GROQ_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


# ============ BM25 SETUP ============

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


# ============ VECTOR (CHROMADB) SETUP ============

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
        contribution = 1 / (k + rank)
        if key not in rrf_scores:
            rrf_scores[key] = {
                "rrf_score": 0.0,
                "text": chunk["text"],
                "source": chunk.get("source_filename"),
                "found_in": [],
            }
        rrf_scores[key]["rrf_score"] += contribution
        rrf_scores[key]["found_in"].append(f"BM25 (rank {rank})")

    for rank, (text, distance, metadata) in enumerate(vector_results, start=1):
        key = get_chunk_key(text)
        contribution = 1 / (k + rank)
        if key not in rrf_scores:
            rrf_scores[key] = {
                "rrf_score": 0.0,
                "text": text,
                "source": metadata.get("source_filename"),
                "found_in": [],
            }
        rrf_scores[key]["rrf_score"] += contribution
        rrf_scores[key]["found_in"].append(f"Vector (rank {rank})")

    return rrf_scores


def hybrid_search(bm25, chunks, collection, embed_model, query, top_n=20, final_k=5):
    bm25_results = search_bm25(bm25, chunks, query, top_k=top_n)
    vector_results = search_vector(collection, embed_model, query, top_k=top_n)
    rrf_scores = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:final_k]


# ============ MAIN: REWRITE -> HYBRID SEARCH ============

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

    print("ChromaDB collection load ho raha hai...\n")
    collection = load_chroma_collection()

    all_results = []

    for q in questions:
        original_query = q["question"]

        # ---- STEP 1: Query rewrite karna ----
        rewritten_query = rewrite_query(original_query)

        print("=" * 80)
        print(f"Q{q['id']} Original : {original_query}")
        print(f"     Rewritten: {rewritten_query}")
        print("=" * 80)

        # ---- STEP 2: Rewritten query se hybrid search karna ----
        hybrid_results = hybrid_search(
            bm25, chunks, collection, embed_model, rewritten_query,
            top_n=TOP_N_PER_METHOD, final_k=FINAL_TOP_K
        )

        top = hybrid_results[0]
        print(f"\n[Hybrid Top Result - RRF]")
        print(f"  RRF Score: {top['rrf_score']:.5f} | Source: {top['source']}")
        print(f"  Found in: {', '.join(top['found_in'])}")
        print(f"  Text: {top['text'][:150]}...")
        print()

        all_results.append({
            "question_id": q["id"],
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "hybrid_top_source": top["source"],
            "hybrid_top_text": top["text"][:200],
            "hybrid_top_rrf_score": top["rrf_score"],
            "hybrid_found_in": top["found_in"],
        })

    with open("hybrid_with_rewrite_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nDone! Results 'hybrid_with_rewrite_results.json' mein save ho gaye hain.")


if __name__ == "__main__":
    main()