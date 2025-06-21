import streamlit as st
import os
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import importlib.util
import sys

# Import existing functions
spec = importlib.util.spec_from_file_location("file_ingestion", "file-ingestion.py")
file_ingestion = importlib.util.module_from_spec(spec)
sys.modules["file_ingestion"] = file_ingestion
spec.loader.exec_module(file_ingestion)
ingest = file_ingestion.ingest

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def initialize_rag():
    """Initialize the RAG components - reusing logic from query.py"""
    try:
        embeddings = FastEmbedEmbeddings()
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        retriever = vectordb.as_retriever()

        # Set up local Mistral model via Ollama with streaming callbacks
        llm = ChatOllama(
            model="mistral",
            streaming=True,  # enable token-level streaming
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        return vectordb, retriever, qa_chain
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        return None, None, None

def process_file_upload(uploaded_file):
    """Process uploaded file and ingest into vector database - reusing ingest function"""
    try:
        # Create documents directory if it doesn't exist
        documents_dir = Path("documents")
        documents_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = documents_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Use the existing ingest function from file-ingestion.py
        ingest(str(file_path))
        
        # Reinitialize RAG components
        st.session_state.vectordb, st.session_state.retriever, st.session_state.qa_chain = initialize_rag()
        
        return True, f"✅ Successfully ingested {uploaded_file.name}"
    except Exception as e:
        return False, f"❌ Error processing file: {e}"

def display_sources(docs):
    """Display source documents with metadata - reusing logic from query.py"""
    if not docs:
        return
    
    st.markdown("### 📚 Sources Used")
    
    sources = {}
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        if source not in sources:
            sources[source] = set()
        sources[source].add(page)
    
    for source, pages in sources.items():
        page_list = sorted(list(pages))
        st.markdown(f"**📄 {source}** (Pages: {', '.join(map(str, page_list))})")
        
        # Show a preview of the content
        with st.expander(f"Preview content from {source}"):
            for doc in docs:
                if doc.metadata.get('source') == source:
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

# Main UI
st.title("🤖 RAG Chat Assistant")
st.markdown("Ask questions about your documents and get AI-powered answers with source citations!")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Document Upload")
    st.markdown("Upload PDF files to add them to the knowledge base.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload & Process"):
            with st.spinner("Processing document..."):
                success, message = process_file_upload(uploaded_file)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    st.markdown("---")
    st.markdown("### 📊 Database Status")
    
    # Check if ChromaDB exists
    if os.path.exists("chroma_db"):
        try:
            vectordb = Chroma(persist_directory="chroma_db", embedding_function=FastEmbedEmbeddings())
            count = vectordb._collection.count()
            st.success(f"✅ Database loaded with {count} documents")
        except:
            st.warning("⚠️ Database exists but may be corrupted")
    else:
        st.info("ℹ️ No database found. Upload documents to get started!")

# Initialize RAG components if not already done
if st.session_state.vectordb is None:
    st.session_state.vectordb, st.session_state.retriever, st.session_state.qa_chain = initialize_rag()

# Chat interface
st.markdown("---")
st.markdown("### 💬 Chat Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            display_sources(message["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if RAG is initialized
    if st.session_state.qa_chain is None:
        st.error("❌ RAG system not initialized. Please upload some documents first.")
    else:
        # Display assistant message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer using the same logic as query.py
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = result.get("result", "Sorry, I couldn't generate an answer.")
                    
                    # Get source documents using the same logic as query.py
                    docs = st.session_state.retriever.get_relevant_documents(prompt)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": docs
                    })
                    
                    # Display sources using the same logic as query.py
                    display_sources(docs)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

# Clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Powered by Ollama + ChromaDB + LangChain*") 