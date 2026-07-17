from retrival import retrieve_chunks
from promptinstruction import build_prompt   # adjust to your actual filename
from LLMcall import call_llm                  # adjust to your actual filename

NO_ANSWER_MSG = "I don't have enough information in the provided documents to answer this."

# ---- Guardrail thresholds (defined here at CLI level, not inside retrieval.py) ----
CONFIDENCE_THRESHOLD = 70.0   # below this -> likely irrelevant
GAP_THRESHOLD = 3.0           # gap between top and weakest result; small gap -> results look too similar, suspicious


def assess_relevance(retrieved: list) -> dict:
    """
    Looks at the retrieved chunks and decides whether they are genuinely relevant
    or likely irrelevant. This acts as a guardrail against hallucination,
    applied before calling the LLM.
    """
    if not retrieved:
        return {"verdict": "NO_RESULTS", "reason": "No chunks were retrieved."}

    confidences = [r["confidence"] for r in retrieved]
    top_conf = confidences[0]
    gap = top_conf - min(confidences)

    if top_conf < CONFIDENCE_THRESHOLD:
        return {
            "verdict": "LIKELY_NOT_RELEVANT",
            "reason": f"Top confidence ({top_conf}%) is below the threshold ({CONFIDENCE_THRESHOLD}%)."
        }

    if gap < GAP_THRESHOLD:
        return {
            "verdict": "UNCERTAIN",
            "reason": f"Gap between top and weakest result is only {round(gap,1)}% — results look too similar to each other."
        }

    return {"verdict": "RELEVANT", "reason": f"Top confidence {top_conf}%, gap {round(gap,1)}% — strong match."}


def answer_question(query: str, top_k: int = 3, verbose: bool = True) -> dict:
    """
    Full RAG pipeline: retrieve -> assess relevance -> (call LLM or refuse directly) -> return result.
    """
    chunks = retrieve_chunks(query, top_k=top_k)
    verdict = assess_relevance(chunks)

    if verbose:
        print(f"\n📊 Retrieval verdict: {verdict['verdict']} — {verdict['reason']}")

    # Guardrail: if the retrieved context looks irrelevant, don't call the LLM at all
    if verdict["verdict"] == "LIKELY_NOT_RELEVANT":
        return {
            "answer": NO_ANSWER_MSG,
            "chunks_used": [],
            "verdict": verdict["verdict"],
            "llm_called": False
        }

    prompt = build_prompt(query, chunks)
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "chunks_used": chunks,
        "verdict": verdict["verdict"],
        "llm_called": True
    }


def run_cli():
    print("=" * 60)
    print("📚 Simple RAG CLI — Ask questions about RLbook.pdf")
    print("Type 'exit' or 'quit' to leave")
    print("=" * 60)

    while True:
        query = input("\n❓ Your question: ").strip()

        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        if not query:
            continue

        result = answer_question(query)

        print("\n" + "-" * 60)
        print(f"💡 ANSWER:\n{result['answer']}")
        if result["llm_called"]:
            sources = ", ".join(f"{c['source']} ({c['page']})" for c in result["chunks_used"])
            print(f"📎 Chunks retrieved from: {sources}")
        print("-" * 60)


if __name__ == "__main__":
    run_cli()