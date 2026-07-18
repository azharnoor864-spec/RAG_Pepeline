# day19_bm25_vs_vector.py
# Step 2: BM25 aur Vector (ChromaDB) search ko 20 questions pe compare karna

import json
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---- Hardcoded paths (apne actual setup ke mutabiq adjust karein) ----
CHUNKS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"
QUESTIONS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day_19\testquestion.json"
CHROMA_DB_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"

# Day 17 mein jo naam collection ko diya tha, wahi yahan daalein
CHROMA_COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

TOP_K = 5


# ============ BM25 SETUP (Day 19 Step 1 se) ============

def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


def search_bm25(bm25, chunks, query, top_k=5):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]


# ============ VECTOR (CHROMADB) SETUP ============

def load_chroma_collection():
    """
    Day 17 mein banaya gaya persistent ChromaDB collection load karte hain.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    return collection


def search_vector(collection, embed_model, query, top_k=5):
    """
    Query ko embedding mein convert karo, phir ChromaDB se
    sabse qareeb (semantically similar) chunks nikaalo.
    """
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # ChromaDB results ko simple list mein convert karna
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return list(zip(documents, distances, metadatas))


# ============ MAIN COMPARISON LOOP ============

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

        # ---- BM25 results ----
        bm25_results = search_bm25(bm25, chunks, query, top_k=TOP_K)
        print("\n[BM25 Top Result]")
        top_chunk, top_score = bm25_results[0]
        print(f"  Score: {top_score:.4f} | Source: {top_chunk['source_filename']}")
        print(f"  Text: {top_chunk['text'][:150]}...")

        # ---- Vector results ----
        vector_results = search_vector(collection, embed_model, query, top_k=TOP_K)
        print("\n[Vector Top Result]")
        top_doc, top_dist, top_meta = vector_results[0]
        print(f"  Distance: {top_dist:.4f} | Source: {top_meta.get('source_filename')}")
        print(f"  Text: {top_doc[:150]}...")

        print()

        # Save for later analysis / report
        all_results.append({
            "question_id": q["id"],
            "question": query,
            "bm25_top_source": top_chunk["source_filename"],
            "bm25_top_text": top_chunk["text"][:200],
            "bm25_top_score": top_score,
            "vector_top_source": top_meta.get("source_filename"),
            "vector_top_text": top_doc[:200],
            "vector_top_distance": top_dist,
        })

    # Save comparison results to a file for the report later
    with open("bm25_vs_vector_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nDone! Results 'bm25_vs_vector_results.json' mein save ho gaye hain.")


if __name__ == "__main__":
    main()