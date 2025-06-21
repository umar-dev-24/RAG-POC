from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    embeddings = FastEmbedEmbeddings()
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    # Set up local Mistral model via Ollama with streaming callbacks
    llm = ChatOllama(
        model="mistral",
        streaming=True,  # enable token-level streaming
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("✅ Ready! Ask questions (type 'exit' to quit)\n")
    
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
            
        print("AI: ", end="", flush=True)

        # Get the answer and retrieve source documents
        result = qa.invoke({"query": q})
        
        # Get the source documents used for the answer
        docs = retriever.get_relevant_documents(q)
        
        # Display source information
        if docs:
            print(f"\n📚 Sources:")
            sources = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                if source not in sources:
                    sources[source] = set()
                sources[source].add(page)
            
            for source, pages in sources.items():
                page_list = sorted(list(pages))
                print(f"   📄 {source} (Pages: {', '.join(map(str, page_list))})")
        
        print()  # newline after completion

if __name__ == "__main__":
    main()
