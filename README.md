# RAG-POC: Retrieval-Augmented Generation with Ollama + ChromaDB

A complete RAG (Retrieval-Augmented Generation) pipeline using Ollama for local LLM inference and ChromaDB for vector storage.

## Features

- 🤖 **Local LLM**: Uses Ollama with Mistral model
- 📚 **Document Ingestion**: PDF processing with metadata tracking
- 🔍 **Vector Search**: ChromaDB for semantic retrieval
- 💬 **Chat Interface**: Both CLI and Web UI options
- 📄 **Source Citations**: Automatic source tracking with page numbers
- 🌐 **Streamlit Web UI**: Beautiful web interface for document upload and chat

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Ollama

Make sure Ollama is running with the Mistral model:

```bash
# Install Mistral model (if not already installed)
ollama pull mistral

# Start Ollama service
ollama serve
```

### 3. Run the Application

#### Option A: Web UI (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

#### Option B: CLI Interface

```bash
# Ingest documents
python file-ingestion.py policy-files/leave-policy.pdf

# Start chat
python query.py
```

## Usage

### Web UI Features

- 📁 **Document Upload**: Drag & drop PDF files in the sidebar
- 💬 **Chat Interface**: Ask questions and get AI-powered answers
- 📚 **Source Citations**: See which documents and pages were used
- 🗑️ **Chat History**: Maintains conversation history with clear option

### CLI Features

- **File Ingestion**: `python file-ingestion.py <path/to/file.pdf>`
- **Interactive Chat**: `python query.py`
- **Demo**: `python demo.py`

## Project Structure

```
RAG-POC/
├── streamlit_app.py      # Web UI application
├── file-ingestion.py     # Document processing and ingestion
├── query.py             # CLI chat interface
├── demo.py              # Simple Ollama demo
├── requirements.txt     # Python dependencies
├── chroma_db/          # Vector database storage
├── documents/          # Uploaded documents (created automatically)
└── policy-files/       # Sample documents
```

## How It Works

1. **Document Processing**: PDFs are loaded, split into chunks, and embedded
2. **Metadata Tracking**: Source filename and page numbers are preserved
3. **Vector Storage**: Chunks are stored in ChromaDB with embeddings
4. **Retrieval**: Semantic search finds relevant document chunks
5. **Generation**: LLM generates answers based on retrieved context
6. **Citation**: Source documents and pages are displayed with answers

## Requirements

- Python 3.8+
- Ollama with Mistral model
- 4GB+ RAM recommended
- Internet connection for initial model download
