import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # .env file se GROQ_API_KEY load karega

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"


def call_llm(prompt: str) -> str:
    """
    Augmented prompt send to Groq LLM to return answer .
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,   # low temperature -> grounded/factual answers, kam creativity
        max_tokens=500
    )
    return response.choices[0].message.content


# ---- Quick test (poora pipeline: retrieve -> prompt -> LLM) ----
if __name__ == "__main__":
    from retrival import retrieve_chunks
    from promptinstruction import build_prompt   

    test_query = "What is Photosynthesis?"

    chunks = retrieve_chunks(test_query, top_k=3)
    final_prompt = build_prompt(test_query, chunks)

    print("Calling LLM...\n")
    answer = call_llm(final_prompt)

    print("=" * 60)
    print(f"QUESTION: {test_query}")
    print("=" * 60)
    print(f"ANSWER:\n{answer}")