import os
import json
import time
import numpy as np
import chromadb
import faiss

# --- Configuration & Cache Resolution ---
CHUNKS_JSON = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"
EMBEDDINGS_CACHE = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\embeddings_cache.npz"

if not os.path.exists(CHUNKS_JSON) or not os.path.exists(EMBEDDINGS_CACHE):
    raise FileNotFoundError("Missing Step 1 cache files! Run your Step 1 benchmark script first.")

# 1. Load data payloads from Step 1 cache
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

texts = [item["text"] for item in chunks_data]
# Standardize keys to string format for ChromaDB dictionary criteria
metadatas = [
    {
        "source_filename": str(item["source_filename"]),
        "page_number": str(item["page_number"]),
        "chunk_index": int(item["chunk_index"])
    } 
    for item in chunks_data
]
ids = [f"id_{idx}" for idx, _ in enumerate(chunks_data)]
cached_vectors = np.load(EMBEDDINGS_CACHE)
model_keys = cached_vectors.files

print(f"Loaded {len(texts)} text chunks and structural metadata files.")
print(f"Detected cached vectors for models: {model_keys}\n")

# --- Step 2: ChromaDB Persistent Storage Configuration ---
print("--- Starting Step 2: ChromaDB Ingestion ---")
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")

chroma_times = {}

for key in model_keys:
    vectors = cached_vectors[key].astype(np.float32)
    # Re-normalize your key strings for clear nomenclature naming schemas
    clean_collection_name = f"coll_{key.replace('/', '_').replace('-', '_')}"
    
    # Reset existing collection instances to guarantee clean benchmarks
    try:
        chroma_client.delete_collection(name=clean_collection_name)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name=clean_collection_name)
    
    t0 = time.time()
    # Batch execute chunk payloads into ChromaDB storage layers
    collection.add(
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    chroma_times[key] = time.time() - t0
    print(f"**ChromaDB**: Ingested {key} | Time: {chroma_times[key]:.2f}s | Items: {collection.count()}")


# --- Step 3: FAISS Pipeline Implementation ---
print("\n--- Starting Step 3: FAISS Index Ingestion ---")

faiss_indices = {}
faiss_times = {}

for key in model_keys:
    vectors = cached_vectors[key].astype(np.float32)
    dimension = vectors.shape[1]
    
    t0 = time.time()
    # Instantiate a clean flat L2 hardware space optimization layer
    index = faiss.IndexFlatL2(dimension)
    
    # FAISS requires native NumPy arrays for vector registration steps
    if not vectors.flags['C_CONTIGUOUS']:
        vectors = np.ascontiguousarray(vectors)
        
    index.add(vectors)
    faiss_times[key] = time.time() - t0
    
    # Retain references in tracking registry arrays
    faiss_indices[key] = index
    print(f"**FAISS**: Built index for {key} | Time: {faiss_times[key]:.4f}s | Count: {index.ntotal}")

# --- Ingestion Benchmark Speed Matrix ---
print(f"\n{'='*60}\nINGESTION PIPELINE VELOCITY MATRIX\n{'='*60}")
print(f"{'Embedding Model Key':<30} {'ChromaDB (s)':<15} {'FAISS (s)':<15}")
for key in model_keys:
    print(f"{key:<30} {chroma_times[key]:<15.3f} {faiss_times[key]:<15.4f}")
