# day19_hybrid_rrf.py
# Step 3: Hybrid Search - BM25 aur Vector ki rankings ko
# Reciprocal Rank Fusion (RRF) se combine karna

import json
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---- Hardcoded paths (apne actual setup ke mutabiq adjust karein) ----
CHUNKS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"
QUESTIONS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day_19\testquestion.json"
CHROMA_DB_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"

CHROMA_COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

TOP_N_PER_METHOD = 20   # har method se kitne candidates lene hain fusion ke liye
RRF_K = 60              # standard smoothing constant
FINAL_TOP_K = 5         # final hybrid list mein kitne chunks dikhane hain


# ============ BM25 SETUP ============

def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def search_bm25(bm25, chunks, query, top_k):
    """
    Returns list of (chunk, score) sorted by score (highest first).
    """
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
    """
    Returns list of (text, distance, metadata) sorted by distance (lowest/best first).
    """
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return list(zip(documents, distances, metadatas))


# ============ RRF FUSION LOGIC (yahi is step ka core hai) ============

def get_chunk_key(text):
    """
    Har chunk ko uniquely identify karne ke liye uska text hi
    key ki tarah use karte hain (simple aur reliable, kyunke
    BM25 aur Vector alag data structures return karte hain).
    """
    return text.strip()


def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    bm25_results: list of (chunk_dict, score) - rank 1 se start hota hai (list ka index+1)
    vector_results: list of (text, distance, metadata) - rank 1 se start hota hai

    Return: dictionary { chunk_key: {"rrf_score": float, "text": str, "source": str, "sources": list} }
    """
    rrf_scores = {}

    # ---- BM25 rankings process karna ----
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

    # ---- Vector rankings process karna ----
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
    """
    Poora hybrid search pipeline: BM25 + Vector -> RRF -> final sorted list
    """
    bm25_results = search_bm25(bm25, chunks, query, top_k=top_n)
    vector_results = search_vector(collection, embed_model, query, top_k=top_n)

    rrf_scores = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)

    # Dictionary ko list mein convert karke RRF score ke hisaab se sort karna
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return sorted_results[:final_k]


# ============ MAIN ============

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
        query = q["question"]
        print("=" * 80)
        print(f"Q{q['id']}: {query}")
        print("=" * 80)

        hybrid_results = hybrid_search(
            bm25, chunks, collection, embed_model, query,
            top_n=TOP_N_PER_METHOD, final_k=FINAL_TOP_K
        )

        print("\n[Hybrid Top Result - RRF]")
        top = hybrid_results[0]
        print(f"  RRF Score: {top['rrf_score']:.5f} | Source: {top['source']}")
        print(f"  Found in: {', '.join(top['found_in'])}")
        print(f"  Text: {top['text'][:150]}...")
        print()

        all_results.append({
            "question_id": q["id"],
            "question": query,
            "hybrid_top_source": top["source"],
            "hybrid_top_text": top["text"][:200],
            "hybrid_top_rrf_score": top["rrf_score"],
            "hybrid_found_in": top["found_in"],
        })

    with open("hybrid_rrf_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nDone! Results 'hybrid_rrf_results.json' mein save ho gaye hain.")


if __name__ == "__main__":
    main()