import os
import json
import time
import numpy as np
import chromadb
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------
# CONFIGURATION: Benchmark Ground Truth Definitions
# ----------------------------------------------------
benchmark_suite = [
    {"query": "What is the core definition of reinforcement learning?", "expected_page_range": range(1, 30)},
    {"query": "How do multi-armed bandits balance exploration and exploitation?", "expected_page_range": range(25, 45)},
    {"query": "Explain the epsilon-greedy action selection strategy.", "expected_page_range": range(25, 45)},
    {"query": "What is an upper confidence bound UCB action selection algorithm?", "expected_page_range": range(35, 50)},
    {"query": "Define the mathematical framework of a Markov Decision Process MDP.", "expected_page_range": range(47, 70)},
    {"query": "What are the components of state transition probabilities?", "expected_page_range": range(47, 70)},
    {"query": "Derive the Bellman Optimality Equation for value functions.", "expected_page_range": range(60, 80)},
    {"query": "How does Dynamic Programming compute optimal policies?", "expected_page_range": range(73, 95)},
    {"query": "Explain the policy evaluation algorithm steps.", "expected_page_range": range(73, 85)},
    {"query": "What is the main difference between policy iteration and value iteration?", "expected_page_range": range(80, 95)},
    {"query": "Define Monte Carlo methods for prediction tasks.", "expected_page_range": range(91, 115)},
    {"query": "How does first-visit Monte Carlo differ from every-visit Monte Carlo?", "expected_page_range": range(91, 110)},
    {"query": "What is temporal-difference TD learning?", "expected_page_range": range(119, 140)},
    {"query": "Explain the step-by-step updates of the SARSA algorithm.", "expected_page_range": range(125, 135)},
    {"query": "What makes Q-learning an off-policy temporal difference method?", "expected_page_range": range(130, 145)},
    {"query": "Describe the n-step TD prediction generalization structure.", "expected_page_range": range(141, 160)},
    {"query": "How do planning and learning interact in the Dyna-Q architecture?", "expected_page_range": range(161, 185)},
    {"query": "What is the purpose of a prioritized sweeping algorithm?", "expected_page_range": range(170, 185)},
    {"query": "Explain policy gradient methods using the policy function parameterization.", "expected_page_range": range(321, 345)},
    {"query": "What is the role of actor-critic methods in reinforcement learning?", "expected_page_range": range(330, 350)}
]

# Load structural mapping metadata cache for FAISS index alignment
with open("day17_chunks.json", "r", encoding="utf-8") as f:
    chunks_metadata = json.load(f)

# Connect to database clients
chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
cached_vectors = np.load("embeddings_cache.npz")
model_keys = cached_vectors.files

print(f"Loaded benchmark script with {len(benchmark_suite)} evaluation targets.")
print(f"Executing retrieval runs over keys: {model_keys}\n")

final_results = {}

for model_key in model_keys:
    print(f"--- Benchmarking Model: {model_key} ---")
    
    # Load model mappings into memory
    hf_path = f"sentence-transformers/{model_key}" if "MiniLM" in model_key or "mpnet" in model_key else "BAAI/bge-large-en-v1.5"
    model = SentenceTransformer(hf_path)
    
    # Resolve vector layout from database layer
    clean_col_name = f"coll_{model_key.replace('/', '_').replace('-', '_')}"
    chroma_col = chroma_client.get_collection(name=clean_col_name)
    
    # Resolve layout from FAISS memory layer
    vectors = cached_vectors[model_key].astype(np.float32)
    vector_dimension = int(vectors.shape[1])
    faiss_index = faiss.IndexFlatL2(vector_dimension)
    faiss_index.add(vectors)
    
    chroma_latencies = []
    faiss_latencies = []
    chroma_precision_scores = []
    faiss_precision_scores = []
    
    # ----------------------------------------------------
    # RUNTIME BENCHMARK EVALUATION LOOP
    # ----------------------------------------------------
    for test in benchmark_suite:
        query_text = test["query"]
        expected_range = test["expected_page_range"]
        
        # 1. Transform raw search query text to math vector format
        query_vector = model.encode(query_text, convert_to_numpy=True).astype(np.float32)
        
        # ---- ChromaDB Query Evaluation ----
        t_start = time.time()
        chroma_res = chroma_col.query(
            query_embeddings=[query_vector.tolist()],
            n_results=3
        )
        chroma_latencies.append(time.time() - t_start)
        
        # Calculate ChromaDB top-3 precision metrics
        chroma_hits = 0
        if chroma_res and chroma_res["metadatas"] and chroma_res["metadatas"][0]:
            for meta in chroma_res["metadatas"][0]:
                # Extract page integer from metadata structure string (e.g. "Page 45" -> 45)
                try:
                    p_num = int(meta["page_number"].split()[-1])
                    if p_num in expected_range:
                        chroma_hits += 1
                except (ValueError, IndexError):
                    continue
        chroma_precision_scores.append(chroma_hits / 3.0)
        
        # ---- FAISS Query Evaluation ----
        # Reshape vector to 2D format required by the FAISS C++ API wrapper
        faiss_query_vector = np.expand_dims(query_vector, axis=0)
        
        t_start = time.time()
        faiss_distances, faiss_indices = faiss_index.search(faiss_query_vector, k=3)
        faiss_latencies.append(time.time() - t_start)
        
        # Calculate FAISS top-3 precision metrics using the index cache map
        faiss_hits = 0
        for idx in faiss_indices[0]:
            if idx < len(chunks_metadata):
                try:
                    p_num = int(chunks_metadata[idx]["page_number"].split()[-1])
                    if p_num in expected_range:
                        faiss_hits += 1
                except (ValueError, IndexError):
                    continue
        faiss_precision_scores.append(faiss_hits / 3.0)

    # Compile aggregate matrix analytics
    final_results[model_key] = {
        "chroma_avg_latency_ms": float(np.mean(chroma_latencies) * 1000),
        "faiss_avg_latency_ms": float(np.mean(faiss_latencies) * 1000),
        "chroma_top3_precision": float(np.mean(chroma_precision_scores)),
        "faiss_top3_precision": float(np.mean(faiss_precision_scores))
    }
    
    print(f"  ChromaDB -> Latency: {final_results[model_key]['chroma_avg_latency_ms']:.2f}ms | Precision@3: {final_results[model_key]['chroma_top3_precision']:.2%}")
    print(f"  FAISS    -> Latency: {final_results[model_key]['faiss_avg_latency_ms']:.2f}ms | Precision@3: {final_results[model_key]['faiss_top3_precision']:.2%}\n")

# --- Step 6: Summary Findings Report ---
print(f"{'='*80}\nFINAL PRODUCTION RAG RETRIEVAL MATRIX\n{'='*80}")
print(f"{'Embedding Model System':<25} | {'DB Type':<10} | {'Latency (ms)':<15} | {'Precision@3':<12}")
print(f"{'-'*80}")
for model_key, metrics in final_results.items():
    print(f"{model_key:<25} | {'ChromaDB':<10} | {metrics['chroma_avg_latency_ms']:<15.2f} | {metrics['chroma_top3_precision']:<12.1%}")
    print(f"{model_key:<25} | {'FAISS':<10} | {metrics['faiss_avg_latency_ms']:<15.2f} | {metrics['faiss_top3_precision']:<12.1%}")
print(f"{'='*80}")
