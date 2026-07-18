# day19_bm25_setup.py
# Step 1: BM25 index banana aur ek simple query test karna
# Chunks structure: source_filename, chunk_index, page_number,
#                    section_heading, chunking_strategy, char_count, text

import json
from rank_bm25 import BM25Okapi

# ---- Hardcoded path (apne actual project folder ke mutabiq adjust karein) ----
CHUNKS_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\outputs\all_chunks.json"


def load_chunks(path):
    """Day 16-18 mein banaye gaye chunks load karte hain."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def tokenize(text):
    """
    BM25 ko words chahiye hote hain (list of tokens), poora sentence nahi.
    Simple tareeqa: lowercase karo, phir space se split karo.
    """
    return text.lower().split()


def build_bm25_index(chunks):
    """
    Har chunk ke 'text' field ko tokenize karke BM25 index banate hain.
    """
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


def search_bm25(bm25, chunks, query, top_k=5):
    """
    Query ko tokenize karo, phir BM25 se scores nikaalo,
    phir top_k sabse zyada score wale chunks return karo.
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Chunks ko score ke hisaab se sort karna (highest score pehle)
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return scored_chunks[:top_k]


if __name__ == "__main__":
    print("Chunks load ho rahe hain...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Total {len(chunks)} chunks load hue.\n")

    print("BM25 index ban raha hai...")
    bm25 = build_bm25_index(chunks)
    print("Index ready hai.\n")

    # Ek test query jo aap ke data se related hai
    test_query = "Supreme Court of Pakistan pendency and disposal"
    print(f"Test query: '{test_query}'\n")

    results = search_bm25(bm25, chunks, test_query, top_k=5)

    print("Top 5 results (BM25 se):\n")
    for i, (chunk, score) in enumerate(results, start=1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   Source: {chunk['source_filename']} | Page: {chunk['page_number']}")
        print(f"   Text: {chunk['text'][:150]}...")
        print()