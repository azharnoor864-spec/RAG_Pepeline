# rag_chain.py
# ============================================
# STEP 2: Retriever + LLM ko jodna
# Ab sirf chunks nahi, ASAL JAWAB milega — source citation ke saath
# ============================================

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from landchain import build_retriever

# .env file se GROQ_API_KEY load karo (agar env variable mein already set
# hai to ye line effect nahi karegi, koi masla nahi)
load_dotenv()


# ---------------------------------------------------------------------------
# Step A: LLM setup — Groq ka llama-3.3-70b-versatile (jo aap Day 14/18 se
# use kar rahi hain)
# ---------------------------------------------------------------------------
def build_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,          # 0 = factual/consistent answers, kam creativity
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Step B: Retrieved chunks ko ek formatted string mein badalna, jisme
# HAR chunk ke sath uska source_filename aur page_number bhi likha ho.
# Ye string hi LLM ko "context" ke tor par diya jayega.
# ---------------------------------------------------------------------------
def format_docs_with_sources(docs) -> str:
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number", "Unknown")
        formatted_chunks.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


# ---------------------------------------------------------------------------
# Step C: Prompt Template — LLM ko batata hai "sirf context se jawab do,
# aur har jawab ke sath source+page zaroor likho"
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions using ONLY the provided context below.

Rules:
1. Answer ONLY using information from the context. If the context doesn't contain
   the answer, say "I don't have enough information in the provided documents to answer this."
2. At the end of your answer, list the sources you used in this exact format:
   Sources: [filename, page X], [filename, page Y]
3. Do not make up information that isn't in the context.

Context:
{context}

Question: {question}

Answer:
""")


# ---------------------------------------------------------------------------
# Step D: Poori Chain banana — retriever + prompt + LLM ko zanjeer (chain)
# ki tarah jodna. Ye LangChain Expression Language (LCEL) syntax hai.
# ---------------------------------------------------------------------------
def build_rag_chain():
    retriever = build_retriever()
    llm = build_llm()

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,  # retrieve -> format
            "question": RunnablePassthrough(),                 # question waise hi pass ho
        }
        | RAG_PROMPT     # dono ko prompt template mein daalo
        | llm            # LLM ko bhejo
        | StrOutputParser()  # LLM ka response object -> plain text nikaalo
    )

    return rag_chain


if __name__ == "__main__":
    print("Building RAG chain...\n")
    chain = build_rag_chain()

    test_question = "what work harry klopf can do?"

    print(f"Question: {test_question}\n")
    print("Generating answer...\n")

    answer = chain.invoke(test_question)

    print("="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)