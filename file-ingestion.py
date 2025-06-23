import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def ingest(pdf_path: str, persist_dir: str = "chroma_db"):
    # 1️⃣ Load the PDF
    docs = PyPDFLoader(pdf_path).load()
    
    # Extract source filename for metadata
    source_filename = os.path.basename(pdf_path)
    
    # 2️⃣ Split into chunks while preserving page numbers
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=50,
        add_start_index=True  # This helps preserve page information
    )
    chunks = splitter.split_documents(docs)
    
    # 3️⃣ Add metadata to each chunk
    for chunk in chunks:
        # Add source filename metadata
        chunk.metadata["source"] = source_filename
        # Ensure page is always an int and starts from 1
        page = chunk.metadata.get("page", 0)
        try:
            page = int(page)
        except Exception:
            page = 0
        chunk.metadata["page"] = page + 1
    
    # 4️⃣ Embed and store with metadata
    embeddings = FastEmbedEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    
    print(f"✅ Ingested '{pdf_path}' -> vector DB '{persist_dir}'.")
    print(f"📄 Processed {len(chunks)} chunks with metadata:")
    print(f"   - Source: {source_filename}")
    print(f"   - Pages: {min(chunk.metadata.get('page', 1) for chunk in chunks)}-{max(chunk.metadata.get('page', 1) for chunk in chunks)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python file-ingestion.py <path/to/your.pdf>")
        sys.exit(1)
    print(f"Processing: {sys.argv[1]}")
    ingest(sys.argv[1])
