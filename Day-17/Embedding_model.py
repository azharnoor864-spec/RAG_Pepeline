import json
import time
import gc
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
CHUNKS_JSON_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"  # Path to your chunking file
BATCH_SIZE = 32                         # Adjust based on your system VRAM/RAM

MODEL_NAME = "BAAI/bge-large-en-v1.5"
HF_PATH = "BAAI/bge-large-en-v1.5"
# ============================================================================

def load_chunks_from_json(json_path):
    """Loads chunks from the provided JSON file path."""
    print(f"Loading chunks from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Successfully loaded {len(chunks)} chunks.")
    return chunks

def process_local_model(model_name, hf_path, texts, batch_size=32):
    """Loads BAAI model and generates embeddings."""
    from sentence_transformers import SentenceTransformer
    
    print(f"\nLoading local model: {model_name} ({hf_path}) ...")
    t0 = time.time()
    model = SentenceTransformer(hf_path)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    t0 = time.time()
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embed_time = time.time() - t0
    
    # Free RAM/VRAM
    del model  
    gc.collect()
    
    metrics = {
        "model": model_name,
        "type": "local (sentence-transformers)",
        "num_chunks": len(texts),
        "embedding_dim": int(embeddings.shape[1]),
        "load_time_sec": round(load_time, 2),
        "embed_time_sec": round(embed_time, 2),
        "chunks_per_sec": round(len(texts) / embed_time, 2) if embed_time > 0 else None,
    }
    return metrics, embeddings

def main():
    if not Path(CHUNKS_JSON_PATH).exists():
        print(f"[ERROR] Chunks file not found: {CHUNKS_JSON_PATH}")
        print("Please ensure day17_chunks.json exists in this directory or update CHUNKS_JSON_PATH.")
        return

    # Step 1: Load text chunks from your file
    chunks = load_chunks_from_json(CHUNKS_JSON_PATH)
    texts = [c["text"] for c in chunks]

    # Step 2: Run embedding generation
    results = []
    embeddings_by_model = {}

    try:
        result, embeddings = process_local_model(MODEL_NAME, HF_PATH, texts, batch_size=BATCH_SIZE)
        results.append(result)
        embeddings_by_model[MODEL_NAME] = embeddings
    except Exception as e:
        print(f"[ERROR] {MODEL_NAME} failed: {e}")
        return

    # Step 3: Save embeddings for downstream vector store steps (ChromaDB/FAISS)
    # Replaces '/' with '_' to ensure clean dictionary keys for npz saving
    save_key = MODEL_NAME.replace("/", "_")
    np.savez("embeddings_cache.npz", **{save_key: embeddings_by_model[MODEL_NAME]})
    print(f"\nSaved embeddings to embeddings_cache.npz (needed for Step 2/3)")

    # Step 4: Summary Output
    print(f"\n{'='*80}\nPROCESSING SUMMARY ({len(texts)} chunks)\n{'='*80}")
    print(f"{'Model':<26} {'Type':<24} {'Dim':<6} {'Load(s)':<9} {'Embed(s)':<10} {'Chunks/sec':<10}")
    for r in results:
        print(f"{r['model']:<26} {r['type']:<24} {r['embedding_dim']:<6} "
              f"{r['load_time_sec']:<9} {r['embed_time_sec']:<10} {r['chunks_per_sec']:<10}")

    with open("embedding_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull metrics saved to embedding_benchmark_results.json")

if __name__ == "__main__":
    main()
