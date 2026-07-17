import chromadb
from sentence_transformers import SentenceTransformer

# ---- Configuration ----
CHROMA_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"
COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# BGE models recommend this prefix for QUERIES only (not for documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- Setup (runs once) ----
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# --- FIX: detect the ACTUAL distance space this collection uses, instead
# of assuming cosine. ChromaDB defaults to "l2" (squared Euclidean) if the
# collection was created WITHOUT an explicit
#   metadata={"hnsw:space": "cosine"}
# argument. Using the cosine-only confidence formula on an l2 collection
# produces meaningless numbers (e.g. a great match showing 40%, or vice
# versa). This reads whatever space was ACTUALLY set, so the formula
# below always matches reality rather than assuming it.
collection_metadata = collection.metadata or {}
DISTANCE_SPACE = collection_metadata.get("hnsw:space", "l2")  # "l2" is Chroma's real default
print(f"Collection distance space detected: '{DISTANCE_SPACE}'")

if DISTANCE_SPACE not in ("cosine", "l2"):
    print(f"  NOTE: space '{DISTANCE_SPACE}' (e.g. 'ip') isn't handled by the "
          f"confidence formula below yet -- raw_distance will still be correct, "
          f"but confidence% will fall back to the l2 formula and may be off.")

print("Loading embedding model (ye thoda time le sakta hai)...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Ready! Collection has {collection.count()} chunks.\n")


def distance_to_confidence(dist: float, space: str) -> float:
    """
    Converts a ChromaDB distance into a 0-100 confidence %, using the
    correct formula for whichever distance space the collection actually
    uses. Both formulas assume embeddings are normalized (unit vectors),
    which is true for sentence-transformers / BGE models by default.

    - "cosine": Chroma DEFINES cosine distance directly as
                (1 - cosine_similarity), range [0, 2].
                So:  cos_sim = 1 - dist

    - "l2":     Chroma's l2 space is SQUARED Euclidean distance, NOT the
                same scale as cosine distance. On unit vectors:
                    ||a - b||^2 = 2 - 2*cos_sim
                So:  cos_sim = 1 - (dist / 2)
                (dist itself ranges [0, 4] here, twice the range of
                cosine distance -- that's WHY the divisor differs
                between the two branches below; using the same formula
                for both, as an earlier version of this function
                mistakenly did, silently produces wrong numbers for one
                of the two spaces.)
    """
    if space == "cosine":
        cos_sim = 1 - dist
    else:  # "l2" (squared Euclidean on normalized vectors) -- and fallback for anything else
        cos_sim = 1 - (dist / 2)

    # map cosine similarity [-1, 1] -> confidence [0, 100]
    confidence = ((cos_sim + 1) / 2) * 100
    return max(0.0, min(100.0, confidence))


def retrieve_chunks(query: str, top_k: int = 5):
    """
    Fetch query from the ChromaDB and top-k most similar chunk.
    Return: list of dicts with text, metadata, aur confidence score
    """
    # Step 1: Query ko embed karo (BGE prefix ke saath)
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = embed_model.encode(prefixed_query).tolist()

    # Step 2: ChromaDB se search karo
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Step 3: Results ko clean format mein convert karo + confidence score
    retrieved = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        confidence = distance_to_confidence(dist, DISTANCE_SPACE)

        retrieved.append({
            "text": doc,
            "source": meta.get("source_filename", "unknown"),
            "page": meta.get("page_number", "unknown"),
            "chunk_index": meta.get("chunk_index"),
            "confidence": round(confidence, 1),
            "raw_distance": round(dist, 4)
        })

    return retrieved


# ---- Quick test ----
if __name__ == "__main__":
    test_query = "what is inline image?"   # apna sample question daal sakte ho
    results = retrieve_chunks(test_query, top_k=3)

    print(f"Query: {test_query}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (confidence: {r['confidence']}%) ---")
        print(f"Source: {r['source']} | Page: {r['page']}")
        print(f"Text: {r['text'][:200]}...")
        print()