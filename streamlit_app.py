import streamlit as st
import os
from pathlib import Path
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
import importlib.util
import sys
from langchain.prompts import PromptTemplate
import difflib

# Define prompt_template at the top level so it is available everywhere
prompt_template = '''
Use ONLY the following context to answer the question. If the answer is not in the context, say "I don't know."
At the end of your answer, cite the source and page in this exact format: [source: filename, page X]

Context:
{context}

Question: {question}
Answer:
'''

# Import existing functions
spec = importlib.util.spec_from_file_location("file_ingestion", "file-ingestion.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load file-ingestion.py module spec.")
file_ingestion = importlib.util.module_from_spec(spec)
sys.modules["file_ingestion"] = file_ingestion
spec.loader.exec_module(file_ingestion)
ingest = file_ingestion.ingest

# Custom streaming callback for Streamlit
class StreamlitStreamingCallback(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

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

if "initialization_attempted" not in st.session_state:
    st.session_state.initialization_attempted = False

def initialize_rag():
    """Initialize the RAG components - reusing logic from query.py, but with strict context prompt."""
    try:
        # Check if database exists and has data
        if not os.path.exists("chroma_db"):
            return None, None, None, "no_database"
            
        embeddings = FastEmbedEmbeddings()
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        
        # Check if database has any documents
        try:
            results = vectordb.get()
            if not results or not results.get('documents') or len(results['documents']) == 0:
                return None, None, None, "empty_database"
        except Exception as e:
            return None, None, None, f"database_error:{str(e)}"
            
        retriever = vectordb.as_retriever()

        # Set up local Phi3 model via Ollama
        llm = ChatOllama(
            model="mistral"
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )
        
        return vectordb, retriever, qa_chain, "success"
    except Exception as e:
        return None, None, None, f"initialization_error:{str(e)}"

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
        st.session_state.initialization_attempted = False  # Reset flag to allow reinitialization
        result = initialize_rag()
        st.session_state.vectordb, st.session_state.retriever, st.session_state.qa_chain, status = result
        
        return True, f"✅ Successfully ingested {uploaded_file.name}"
    except Exception as e:
        return False, f"❌ Error processing file: {e}"

# Main UI
st.title("🤖 RAG Chat")
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
    
    # Check if ChromaDB exists and is properly initialized
    if os.path.exists("chroma_db"):
        try:
            # Initialize embeddings
            embeddings = FastEmbedEmbeddings()
            vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
            
            # Get collection count - try different methods
            try:
                # Method 1: Try to get count from collection
                count = vectordb._collection.count()
                st.success(f"✅ Database loaded with {count} chunks")
            except AttributeError:
                try:
                    # Method 2: Try to get count using get method
                    results = vectordb.get()
                    count = len(results['documents']) if results and 'documents' in results else 0
                    st.success(f"✅ Database loaded with {count} chunks")
                except Exception:
                    # Method 3: Just indicate database exists
                    st.success("✅ Database loaded successfully")
                    
        except Exception as e:
            st.warning(f"⚠️ Database exists but may be corrupted: {str(e)}")
    else:
        st.info("ℹ️ No database found. Upload documents to get started!")

# Initialize RAG components if not already done
if st.session_state.vectordb is None and not st.session_state.initialization_attempted:
    st.session_state.initialization_attempted = True
    with st.spinner("🔄 Initializing RAG system..."):
        result = initialize_rag()
        st.session_state.vectordb, st.session_state.retriever, st.session_state.qa_chain, status = result
        
        # Handle different status messages
        if status == "no_database":
            st.info("ℹ️ No database found. Please upload documents to get started with the RAG system.")
        elif status == "empty_database":
            st.info("ℹ️ Database exists but is empty. Please upload documents to get started with the RAG system.")
        elif status.startswith("database_error:"):
            st.warning(f"⚠️ Database error: {status.split(':', 1)[1]}")
        elif status.startswith("initialization_error:"):
            st.error(f"❌ Initialization error: {status.split(':', 1)[1]}")
        elif status == "success":
            st.success("✅ RAG system initialized successfully!")

# Chat interface
st.markdown("---")
st.markdown("### 💬 Chat Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if st.session_state.qa_chain is None:
        st.error("❌ RAG system not initialized. Please upload some documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_container = st.empty()
                    streaming_callback = StreamlitStreamingCallback(response_container)
                    streaming_llm = ChatOllama(
                        model="mistral",
                        callbacks=[streaming_callback]
                    )
                    prompt_obj = PromptTemplate(
                        input_variables=["context", "question"],
                        template=prompt_template,
                    )
                    temp_qa_chain = RetrievalQA.from_chain_type(
                        llm=streaming_llm,
                        retriever=st.session_state.retriever,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": prompt_obj}
                    )
                    result = temp_qa_chain.invoke({"query": prompt})
                    answer = result.get("result", "Sorry, I couldn't generate an answer.")
                    response_container.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
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