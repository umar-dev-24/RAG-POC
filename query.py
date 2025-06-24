from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main():
    # Initialize embedding and vector DB
    embeddings = FastEmbedEmbeddings()
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever( search_type="mmr",  # or "similarity" (default)
    search_kwargs={
        "k": 3,                # number of top chunks to return
    })

    # Set up local Mistral model via Ollama with streaming callback
    llm = ChatOllama(
        model="mistral",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the question. If you don't know the answer, just say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
    )

    print("✅ Ready! Ask questions (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # 1. Get relevant documents
        docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Format the prompt
        final_prompt = prompt_template.format(context=context, question=q)

        # 3. Print the full prompt for debugging
        print("\n📤 Prompt sent to LLM:\n" + "-"*50)
        print(final_prompt[:2000])  # Avoid printing very long prompts
        print("-"*50 + "\n")

        # 4. Send to LLM (streaming output will appear)
        print("AI: ", end="", flush=True)
        llm.invoke(final_prompt)
        print()  # newline after answer

if __name__ == "__main__":
    main()
