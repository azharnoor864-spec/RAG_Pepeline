# langchain_retriever_setup.py
# ============================================
# LangChain Retriever jo aapke EXISTING Day 17 ChromaDB se connect karta hai
# ============================================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, TOP_K


def build_retriever():
    """
    Existing ChromaDB collection ko LangChain retriever mein wrap karta hai.
    Koi naya data nahi banta - Day 17 ka data hi reuse hota hai.
    """
    # Step 1: Same embedding model wrap karo jo Day 17 mein collection banate
    # waqt use hua tha. Ye MATCH hona zaroori hai, warna query embedding
    # aur stored embeddings ka dimension/space mismatch ho jayega.
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Step 2: Existing persistent ChromaDB collection se connect karo
    # (Ye NAYA collection nahi bana raha - existing wala load kar raha hai)
    print(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Sanity check - collection mein kitne documents hain
    try:
        count = vectorstore._collection.count()
        print(f"Collection mein total chunks: {count}")
        if count == 0:
            print("WARNING: Collection empty hai! Path ya collection name check karein.")
    except Exception as e:
        print(f"Collection count check karne mein error: {e}")

    # Step 3: Ise ek Retriever object mein convert karo
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return retriever


def test_retriever(retriever, query: str):
    """Ek test query chala kar retrieved chunks dikhata hai."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    results = retriever.invoke(query)

    print(f"\nRetrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source_filename', 'N/A')
        page = doc.metadata.get('page_number', 'N/A')
        print(f"--- Chunk {i} ---")
        print(f"Source_filename: {source}")
        print(f"Page_number: {page}")
        print(f"Content preview: {doc.page_content}...")
        print()


if __name__ == "__main__":
    retriever = build_retriever()

    test_query = "What working can Harry klopfs do"
    test_retriever(retriever, test_query)