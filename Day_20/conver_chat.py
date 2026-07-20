# conversational_rag_chain.py
# ============================================
# STEP 2: Conversational RAG - Follow-up questions ke saath context na khoye
# ============================================

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from landchain import build_retriever

load_dotenv()


def build_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# STEP A: Contextualize Prompt
# Ye prompt LLM ko batata hai: "History dekho, naya ambiguous sawaal lo,
# aur ise ek STANDALONE (apne aap mein complete) sawaal mein badal do."
# IMPORTANT: Ye khud jawab NAHI deta - sirf sawaal ko reformulate karta hai.
# ---------------------------------------------------------------------------
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference "
     "context in the chat history, formulate a standalone question which can be "
     "understood without the chat history. Do NOT answer the question, just "
     "reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ---------------------------------------------------------------------------
# STEP B: Answer Prompt
# Ye wahi answer-generation prompt hai (jaisa rag_chain.py mein tha), lekin
# ab isme chat_history bhi shamil hai taake LLM tone/context samjhe.
# ---------------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions using ONLY the provided "
     "context below.\n\n"
     "STRICT RULES:\n"
     "1. Answer using ONLY facts explicitly stated in the context. Do NOT add "
     "any detail (names, places, institutions, dates) that is not literally "
     "present in the context, even if it seems plausible.\n"
     "2. If a name of an organization/institution is not written verbatim in "
     "the context, do NOT mention any organization/institution name at all - "
     "just omit that detail rather than guessing.\n"
     "3. If the context does not mention a specific detail, say so - do not "
     "guess or fill in gaps.\n"
     "4. Do NOT copy any '[Source: ...]' markers from the context into your "
     "answer text - those are for your reference only, not for the response.\n"
     "5. Do NOT invent additional questions or a Q&A format. Just answer the "
     "single question asked, directly, in plain prose, as if you were "
     "writing one paragraph.\n"
     "6. At the very end, add ONE line: Sources: [filename, page X].\n\n"
     "EXAMPLE of correct behavior:\n"
     "Context: 'Trained in neurophysiology, Harry was a senior scientist "
     "affiliated with the Avionics Directorate of AFOSR at Wright-Patterson "
     "Air Force Base.'\n"
     "Question: What was his background?\n"
     "Correct answer: 'Harry was trained in neurophysiology and was a senior "
     "scientist affiliated with the Avionics Directorate of the Air Force "
     "Office of Scientific Research (AFOSR) at Wright-Patterson Air Force "
     "Base.\\n\\nSources: [RLbook.pdf, page 15]'\n"
     "(Notice: the answer copies the institution name EXACTLY as written in "
     "the context - it does not substitute a different, more familiar-"
     "sounding institution name.)\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# IMPORTANT: create_stuff_documents_chain by DEFAULT sirf doc.page_content
# ko jama karta hai - metadata (source_filename, page_number) ko IGNORE
# kar deta hai. Isi wajah se LLM ko sahi source pata nahi chalta tha aur
# wo fake/hallucinated source bana raha tha. Ye document_prompt batata hai
# ke HAR chunk ko metadata ke saath kaise format karna hai.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {source_filename} | Page: {page_number}]\n{page_content}"
)


def build_conversational_rag_chain():
    retriever = build_retriever()
    llm = build_llm()

    # Step A ka chain: retriever ko "history-aware" bana dete hain.
    # Ab retriever.invoke() se pehle, LLM khud sawaal reformulate karega.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )

    # Step B ka chain: retrieved chunks ko "stuff" (seedha jama) kar ke
    # ANSWER_PROMPT mein daal dete hain, phir LLM ko bhejte hain.
    # document_prompt=DOCUMENT_PROMPT -> ab har chunk ke sath uska
    # source_filename aur page_number bhi LLM ko dikhega
    question_answer_chain = create_stuff_documents_chain(
        llm, ANSWER_PROMPT, document_prompt=DOCUMENT_PROMPT
    )

    # Dono ko jodna: history_aware_retriever -> question_answer_chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def build_conversational_compressed_chain():
    """
    FINAL VERSION - CrossEncoderReranker use kar rahe hain.

    History: LLMChainExtractor (fake facts), LLMChainFilter (fake facts +
    crash), EmbeddingsFilter (safe lekin BGE cosine similarity "cliff
    effect" ki wajah se discrimination weak thi - 0.75 pe sab pass, 0.85
    pe sab reject).

    CrossEncoderReranker query aur chunk DONO ko ek sath ek model mein
    daalta hai (na ke alag-alag embeddings compare karta hai), isliye
    zyada accurate score deta hai aur BGE ka cliff-effect masla nahi
    hota. Koi LLM call bhi nahi - isliye crash/hallucination ka risk nahi.
    """
    retriever = build_retriever()
    llm = build_llm()

    # Cross-encoder model load karo - ye query+chunk dono ko sath dekh
    # kar relevance score deta hai
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    # top_n=3 -> retrieved chunks mein se sirf top 3 (sabse relevant)
    # rakho, baaki hata do
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )

    # Ab history-aware wrapper, COMPRESSED retriever ke upar lagao
    # (normal retriever ki jagah)
    history_aware_compressed_retriever = create_history_aware_retriever(
        llm, compressed_retriever, CONTEXTUALIZE_PROMPT
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, ANSWER_PROMPT, document_prompt=DOCUMENT_PROMPT
    )

    final_chain = create_retrieval_chain(
        history_aware_compressed_retriever, question_answer_chain
    )

    return final_chain


if __name__ == "__main__":
    llm = build_llm()
    chain = build_conversational_rag_chain()

    # Chat history manually maintain kar rahe hain yahan (list of messages)
    # FastAPI step mein ye session-based ban jayega (Step 5)
    chat_history = []

    def ask(question: str):
        print(f"\nUser: {question}")

        # DEBUG: dekho LLM ne sawaal ko kis standalone form mein badla
        # (history_aware_retriever ke andar ye chupa hua hota hai, isliye
        # hum contextualize step ko alag se bhi call kar ke dikha rahe hain)
        reformulated = llm.invoke(
            CONTEXTUALIZE_PROMPT.format_messages(
                chat_history=chat_history, input=question
            )
        ).content
        print(f"[DEBUG] Reformulated question: {reformulated!r}")

        result = chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        # DEBUG: dekho retriever ne asal mein kaunse chunks diye - taake
        # pata chale answer kis text se aaya
        print("\n--- DEBUG: Retrieved chunks ---")
        for i, doc in enumerate(result.get("context", []), 1):
            src = doc.metadata.get("source_filename", "N/A")
            pg = doc.metadata.get("page_number", "N/A")
            print(f"  [{i}] {src} | page {pg} | {doc.page_content[:100]!r}")
        print("--- END DEBUG ---\n")

        answer = result["answer"]
        print(f"Assistant: {answer}")

        # History update karo - agla sawaal isi context ko use karega
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        return answer

    # Test: pehle sawaal, phir 2 follow-ups - jaisa Noor ne test kiya
    ask("what can do harry klopf?")
    ask("what is the supreme court quality?")
    ask("what can work do?")