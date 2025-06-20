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

        # Stream output via callback
        result = qa.invoke({"query": q})  # the actual answer will stream automatically

        print()  # newline after completion

if __name__ == "__main__":
    main()
