# day19_query_rewrite.py
# Step 4: Query Rewriting - LLM se user ka sawal
# behtar retrieval query mein rewrite karwana

import os
from groq import Groq
from dotenv import load_dotenv
# ---- Groq client setup ----
# GROQ_API_KEY environment variable mein set hona chahiye
load_dotenv()
client = Groq()

MODEL_NAME = "llama-3.3-70b-versatile"

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


def rewrite_query(original_query):
    """
    User ke original sawal ko LLM se behtar retrieval query mein rewrite karwate hain.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": original_query},
        ],
        temperature=0.2,   # low temperature -> consistent, non-creative rewriting
        max_tokens=100,
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten


if __name__ == "__main__":
    test_queries = [
        "disposal rate kya hai?",
        "how does it wrap around images",
        "what powers council has",
    ]

    for q in test_queries:
        rewritten = rewrite_query(q)
        print(f"Original : {q}")
        print(f"Rewritten: {rewritten}")
        print("-" * 60)