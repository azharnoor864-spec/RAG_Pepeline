# main.py
# ============================================
# MAIN ENTRY POINT - Ye file khud koi RAG logic nahi likhti.
# Ye sirf conversational_rag_chain.py (aapki conver_chat.py) se function
# IMPORT/CALL karti hai, aur usay FastAPI ke andar wrap kar deti hai.
#
# Import chain:
#   rag_config.py -> langchain_retriever_setup.py -> conversational_rag_chain.py -> main.py
# ============================================

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# Doosri file se function call kar rahe hain - code duplicate nahi kiya
# NOTE: agar aapki file ka naam conver_chat.py hai, to yahan
# "conversational_rag_chain" ki jagah "conver_chat" likhein
from conver_chat import build_conversational_compressed_chain

app = FastAPI(title="Conversational RAG API")

# Chain ek hi baar bante hai (server start hote waqt)
print("Building RAG chain, please wait...")
rag_chain = build_conversational_compressed_chain()
print("RAG chain ready.")

# Session memory: { session_id: [chat_history] }
session_store: dict[str, list] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    debug_context: str = ""  # TEMPORARY - taake hum dekh sakein LLM ko kya mila


@app.post("/api/rag/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    chat_history = session_store.get(request.session_id, [])

    try:
        result = rag_chain.invoke({
            "input": request.message,
            "chat_history": chat_history,
        })
    except Exception as e:
        # LLM kabhi kabhi format/parsing issues de sakta hai (jaisa humne
        # LLMChainFilter ke sath dekha) - poora server crash hone ki
        # bajaye, clean error message wapas bhejo
        return ChatResponse(
            answer=f"Sorry, an error occurred while generating the answer: {str(e)}",
            sources=[],
            session_id=request.session_id,
            debug_context="",
        )

    answer = result["answer"]

    sources = []
    for doc in result.get("context", []):
        sources.append({
            "filename": doc.metadata.get("source_filename", "Unknown"),
            "page": doc.metadata.get("page_number", "Unknown"),
        })

    chat_history.append(HumanMessage(content=request.message))
    chat_history.append(AIMessage(content=answer))
    session_store[request.session_id] = chat_history

    # TEMPORARY DEBUG: poora context text jo LLM ko bheja gaya, taake
    # pata chale ke retrieval/compression sahi tha ya answer-generation
    # mein masla tha
    debug_text = "\n\n".join(
        f"[{d.metadata.get('source_filename')} p{d.metadata.get('page_number')}]\n{d.page_content}"
        for d in result.get("context", [])
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=request.session_id,
        debug_context=debug_text,
    )


@app.get("/api/rag/history/{session_id}")
def get_history(session_id: str):
    """Debug ke liye - kisi session ki poori history dekhne ka endpoint"""
    chat_history = session_store.get(session_id, [])
    return {
        "session_id": session_id,
        "turns": [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in chat_history
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)