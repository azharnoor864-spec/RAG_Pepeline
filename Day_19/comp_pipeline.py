# day19_full_pipeline.py
# =========================================================================
# COMPLETE ADVANCED RAG PIPELINE (Day 19)
#
# Flow: User Query
#         -> Query Rewriting (Groq LLM)
#         -> Hybrid Search (BM25 + Vector + RRF)
#         -> Cross-Encoder Re-ranking
#         -> Hierarchical Parent Expansion (page-based, window fallback)
#         -> Relevance Guardrail (Day 18 se concept)
#         -> Final Answer Generation (Groq LLM)
# =========================================================================

import os
import json
import chromadb
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------
# HARDCODED PATHS (apne actual setup ke mutabiq adjust karein)
# ---------------------------------------------------------------------
CHUNKS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"
CHROMA_DB_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"

CHROMA_COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------
# PIPELINE SETTINGS (yahan se tune karein)
# ---------------------------------------------------------------------
TOP_N_PER_METHOD = 20     # BM25 aur Vector se kitne candidates lene hain
RRF_K = 60                # RRF smoothing constant
HYBRID_TOP_K = 20         # RRF ke baad re-ranker ko kitne candidates dene hain
FINAL_TOP_K = 3           # re-ranking ke baad final kitne chunks LLM ko dene hain

MAX_PARENT_CHARS = 2400   # ~600 tokens (4 chars ~ 1 token approx) - is se zyada bada parent nahi banega
WINDOW_SIZE = 1           # window fallback mein kitne neighbors (aage/peeche) lene hain

RELEVANCE_THRESHOLD = 3.0  # cross-encoder score is se kam ho to "not relevant" treat karo

groq_client = Groq()


# =========================================================================
# STEP 0: DATA LOADING + INDEX BUILDING (ek dafa startup pe hota hai)
# =========================================================================

def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def build_parent_lookup(chunks):
    """
    Hierarchical RAG ke liye do cheezein taiyar karta hai:

    1. page_lookup: (source_filename, page_number) -> combined page text
       (Approach A: metadata/page-based parent)

    2. ordered_by_source: source_filename -> chunk_index ke hisaab se sorted chunks
       (Approach B: window expansion fallback ke liye)
    """
    page_groups = {}
    ordered_by_source = {}

    for c in chunks:
        src = c.get("source_filename", "unknown")
        page = c.get("page_number")

        # ---- Page-based grouping ----
        key = (src, page)
        if key not in page_groups:
            page_groups[key] = []
        page_groups[key].append(c)

        # ---- Source-based ordering (window ke liye) ----
        if src not in ordered_by_source:
            ordered_by_source[src] = []
        ordered_by_source[src].append(c)

    # Har page group ke chunks ko chunk_index ke hisaab se sort + combine karna
    page_lookup = {}
    for key, group_chunks in page_groups.items():
        sorted_group = sorted(group_chunks, key=lambda x: x.get("chunk_index", 0))
        combined_text = " ".join(c["text"] for c in sorted_group)
        page_lookup[key] = combined_text

    # Har source ke chunks ko chunk_index ke hisaab se sort karna (window ke liye)
    for src in ordered_by_source:
        ordered_by_source[src].sort(key=lambda x: x.get("chunk_index", 0))

    return page_lookup, ordered_by_source


# =========================================================================
# STEP 1: QUERY REWRITING
# =========================================================================

REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for a retrieval system.
Rewrite the user's question into a clearer, more specific version for document search.

Rules:
- Keep the rewritten query factual and neutral, do not answer the question.
- Keep important keywords (names, numbers, dates, technical terms) intact.
- Do NOT add facts, countries, dates, or details not implied by the original question.
- Output ONLY the rewritten query, nothing else.
"""


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


# =========================================================================
# STEP 2: HYBRID SEARCH (BM25 + VECTOR + RRF)
# =========================================================================

def search_bm25(bm25, chunks, query, top_k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def search_vector(collection, embed_model, query, top_k):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return list(zip(results["documents"][0], results["distances"][0], results["metadatas"][0]))


def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    rrf_scores = {}

    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        key = chunk["text"].strip()
        if key not in rrf_scores:
            rrf_scores[key] = {
                "rrf_score": 0.0, "text": chunk["text"],
                "source": chunk.get("source_filename"),
                "page_number": chunk.get("page_number"),
                "chunk_index": chunk.get("chunk_index"),
                "found_in": [],
            }
        rrf_scores[key]["rrf_score"] += 1 / (k + rank)
        rrf_scores[key]["found_in"].append(f"BM25(rank {rank})")

    for rank, (text, distance, metadata) in enumerate(vector_results, start=1):
        key = text.strip()
        if key not in rrf_scores:
            rrf_scores[key] = {
                "rrf_score": 0.0, "text": text,
                "source": metadata.get("source_filename"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "found_in": [],
            }
        rrf_scores[key]["rrf_score"] += 1 / (k + rank)
        rrf_scores[key]["found_in"].append(f"Vector(rank {rank})")

    return rrf_scores


def hybrid_search(bm25, chunks, collection, embed_model, query, top_n, final_k):
    bm25_results = search_bm25(bm25, chunks, query, top_k=top_n)
    vector_results = search_vector(collection, embed_model, query, top_k=top_n)
    rrf_scores = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:final_k]


# =========================================================================
# STEP 3: CROSS-ENCODER RE-RANKING
# =========================================================================

def rerank_with_cross_encoder(cross_encoder, query, candidates, top_k):
    pairs = [[query, c["text"]] for c in candidates]
    ce_scores = cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, ce_scores):
        candidate["cross_encoder_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked[:top_k]


# =========================================================================
# STEP 4: HIERARCHICAL PARENT EXPANSION
# =========================================================================

def get_window_expansion(source, chunk_index, ordered_by_source, window_size=1):
    """
    Approach B: neighboring chunks (chunk_index se aage/peeche) combine karna.
    """
    source_chunks = ordered_by_source.get(source, [])

    # us chunk ki position dhoondo list mein (chunk_index se match karke)
    position = None
    for i, c in enumerate(source_chunks):
        if c.get("chunk_index") == chunk_index:
            position = i
            break

    if position is None:
        return None

    start = max(0, position - window_size)
    end = min(len(source_chunks), position + window_size + 1)
    window_chunks = source_chunks[start:end]

    return " ".join(c["text"] for c in window_chunks)


def expand_to_parent(candidate, page_lookup, ordered_by_source):
    """
    Hybrid logic (jaisa humne discuss kiya):
    1. Pehle page-based parent try karo.
    2. Agar parent MAX_PARENT_CHARS se zyada bada ho, truncate karo.
    3. Agar page-based parent chota/missing ho (jaise fragmented docx chunks),
       window expansion pe fallback karo.
    """
    source = candidate["source"]
    page = candidate.get("page_number")
    chunk_index = candidate.get("chunk_index")

    parent_text = None

    # ---- Approach A: page-based parent ----
    key = (source, page)
    if page is not None and key in page_lookup:
        parent_text = page_lookup[key]

    # ---- Fallback: agar page parent missing ya bohat chota hai ----
    if not parent_text or len(parent_text) < len(candidate["text"]) * 1.2:
        window_text = get_window_expansion(source, chunk_index, ordered_by_source, WINDOW_SIZE)
        if window_text and len(window_text) > len(parent_text or ""):
            parent_text = window_text

    if not parent_text:
        parent_text = candidate["text"]  # last resort: original chunk hi rakho

    # ---- Token-budget safety: bohat bada parent truncate karna ----
    if len(parent_text) > MAX_PARENT_CHARS:
        parent_text = parent_text[:MAX_PARENT_CHARS] + " ...[truncated]"

    candidate["parent_text"] = parent_text
    candidate["parent_char_count"] = len(parent_text)
    candidate["approx_tokens"] = len(parent_text) // 4  # rough approximation

    return candidate


# =========================================================================
# STEP 5: RELEVANCE GUARDRAIL (Day 18 se concept, cross-encoder score se)
# =========================================================================

def assess_relevance(top_candidates, threshold=RELEVANCE_THRESHOLD):
    """
    Agar sabse best candidate ka cross-encoder score bhi threshold se kam hai,
    matlab retrieval ne kuch "confidently relevant" nahi paaya -> hallucination
    se bachne ke liye LLM ko batayenge ke context weak hai.
    """
    if not top_candidates:
        return False
    best_score = top_candidates[0]["cross_encoder_score"]
    return best_score >= threshold


# =========================================================================
# STEP 6: FINAL ANSWER GENERATION
# =========================================================================

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context.

Rules:
- Answer using only the information in the context below.
- If the context does not contain enough information to answer confidently, say so clearly.
- Do not make up facts, numbers, or details not present in the context.
- Cite which source document your answer comes from when relevant.
"""


def generate_answer(query, expanded_candidates, is_relevant):
    if not is_relevant:
        context_note = ("\n[NOTE: Retrieval confidence is LOW for this query. "
                         "The context below may not be strongly relevant. "
                         "Be cautious and say so if unsure.]\n")
    else:
        context_note = ""

    context_blocks = []
    for i, c in enumerate(expanded_candidates, start=1):
        context_blocks.append(
            f"[Source {i}: {c['source']} | Page: {c.get('page_number')}]\n{c['parent_text']}"
        )
    context_text = "\n\n".join(context_blocks)

    user_prompt = f"{context_note}\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# =========================================================================
# MAIN PIPELINE (ek query ke liye pura flow)
# =========================================================================

def run_pipeline(query, bm25, chunks, collection, embed_model, cross_encoder,
                  page_lookup, ordered_by_source, verbose=True):

    # ---- Step 1: Query Rewriting ----
    rewritten = rewrite_query(query)
    if verbose:
        print(f"\n[Rewritten Query] {rewritten}")

    # ---- Step 2: Hybrid Search ----
    hybrid_candidates = hybrid_search(
        bm25, chunks, collection, embed_model, rewritten,
        top_n=TOP_N_PER_METHOD, final_k=HYBRID_TOP_K
    )

    # ---- Step 3: Cross-Encoder Re-ranking ----
    reranked = rerank_with_cross_encoder(cross_encoder, rewritten, hybrid_candidates, FINAL_TOP_K)

    # ---- Step 4: Hierarchical Parent Expansion ----
    expanded = [expand_to_parent(c, page_lookup, ordered_by_source) for c in reranked]

    if verbose:
        print("\n[Top Retrieved Chunks After Full Pipeline]")
        for i, c in enumerate(expanded, start=1):
            print(f"  {i}. CE Score: {c['cross_encoder_score']:.3f} | "
                  f"Source: {c['source']} | ~{c['approx_tokens']} tokens")

    # ---- Step 5: Relevance Guardrail ----
    is_relevant = assess_relevance(expanded)
    if verbose and not is_relevant:
        print("\n[Warning] Retrieval confidence LOW - answer may be uncertain.")

    # ---- Step 6: Final Answer Generation ----
    answer = generate_answer(query, expanded, is_relevant)

    return answer, expanded


# =========================================================================
# CLI - REAL-TIME USER QUERY LOOP
# =========================================================================

def main():
    print("Setup run...\n")

    chunks = load_chunks()
    print(f"  {len(chunks)} chunks loaded.")

    bm25 = build_bm25_index(chunks)
    print("  BM25 index ready.")

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("  Embedding model ready.")

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print("  Cross-encoder ready.")

    collection = load_chroma_collection()
    print("  ChromaDB collection ready.")

    page_lookup, ordered_by_source = build_parent_lookup(chunks)
    print("  Hierarchical parent lookup ready.\n")

    print("=" * 70)
    print("Advanced RAG Pipeline Ready! Question ('exit' for end conversation)")
    print("=" * 70)

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Allah Hafiz!")
            break
        if not query:
            continue

        answer, expanded = run_pipeline(
            query, bm25, chunks, collection, embed_model, cross_encoder,
            page_lookup, ordered_by_source, verbose=True
        )

        print("\n" + "-" * 70)
        print("ANSWER:")
        print(answer)
        print("-" * 70)


if __name__ == "__main__":
    main()