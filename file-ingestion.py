from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def ingest(pdf_path: str, persist_dir: str = "chroma_db"):
    # 1️⃣ Load the PDF
    docs = PyPDFLoader(pdf_path).load()
    # 2️⃣ Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # 3️⃣ Embed and store
    embeddings = FastEmbedEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"✅ Ingested '{pdf_path}' -> vector DB '{persist_dir}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path/to/your.pdf>")
        sys.exit(1)
    print(sys.argv)
    ingest(sys.argv[1])
