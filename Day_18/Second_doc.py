import os
import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber   

# ---- Configuration ----
CHROMA_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\Day-17\chroma_db_storage"
COLLECTION_NAME = "coll_BAAI_bge_large_en_v1.5"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

NEW_PDF_PATH = r"C:\Users\PMYLS\Desktop\RAG\enterprise_rag_engine\files\MACHINE LEARNING.pdf"   
NEW_SOURCE_NAME = "MACHINE LEARNING.pdf"          # metadata save 

CHUNK_SIZE = 500      # characters per chunk 
CHUNK_OVERLAP = 50    # overlap context 

# ---- Step 1: Extract text from PDF (page-by-page,  page_number metadata ----
def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_number": i, "text": text})
    return pages


# ---- Step 2: Chunk each page's text ----
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---- Main ingestion ----
def main():
    print(f"Extracting text from {NEW_PDF_PATH}...")
    pages = extract_pages(NEW_PDF_PATH)

    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"])
        for c in page_chunks:
            if c.strip():   # skip empty chunks
                all_chunks.append({"text": c, "page_number": page["page_number"]})

    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages.")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()

    # Unique IDs, prefixed with source name to avoid clashing with RLbook's ids
    ids = [f"seconddoc_{i}" for i in range(len(all_chunks))]
    metadatas = [
        {
            "source_filename": NEW_SOURCE_NAME,
            "page_number": str(c["page_number"]),
            "chunk_index": i
        }
        for i, c in enumerate(all_chunks)
    ]

    print("Connecting to ChromaDB and adding to collection...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Done! Collection now has {collection.count()} total chunks.")


if __name__ == "__main__":
    main()