def build_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Retrieved chunks and  user query to make structured prompt and combine combine it,
    to send LLM.
    """
    if not retrieved_chunks:
        context_block = "No relevant context was found in the document collection."
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}, {chunk['page']} "
                f"(confidence: {chunk['confidence']}%)]\n{chunk['text']}"
            )
        context_block = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided context below.

RULES:
1. Only use information from the CONTEXT section to answer. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, respond exactly with: "I don't have enough information in the provided documents to answer this."
3. When you answer, cite the source(s) you used, like this: (Source: <filename>, <page>).
4. Do not fabricate page numbers, sources, or facts that are not present in the context.

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""

    return prompt


# ---- Quick test (retrieval.py se chunks lekar) ----
if __name__ == "__main__":
    from retrival import retrieve_chunks  

    test_query = "What is Photosynthesis?"
    chunks = retrieve_chunks(test_query, top_k=3)
    final_prompt = build_prompt(test_query, chunks)

    print("=" * 60)
    print("GENERATED PROMPT:")
    print("=" * 60)
    print(final_prompt)