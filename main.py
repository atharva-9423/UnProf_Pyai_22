import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("=" * 60)
    print("  Day 22: Context-Aware Conversational RAG CLI")
    print("=" * 60)

    docs_folder = "documents"
    if not os.path.exists(docs_folder):
        print(f"Error: '{docs_folder}' folder not found.")
        sys.exit(1)

    print("[1/4] Loading documents...")
    loader = DirectoryLoader(docs_folder, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    if not docs:
        print(f"Error: No .txt files found in '{docs_folder}'.")
        sys.exit(1)

    print("[2/4] Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print("[3/4] Creating embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("[4/4] Initializing LLM and Retrieval Chain with Memory...")
    if "GEMINI_API_KEY" not in os.environ:
        print("\nWARNING: GEMINI_API_KEY environment variable not set.")
        print("Please set it to use the LLM for answer generation.")
        print("Example: $env:GEMINI_API_KEY=\"your_api_key\"\n")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash", 
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    def get_contextualized_question(inputs):
        history = inputs.get("chat_history", [])
        if not history:
            return inputs["input"]
        
        last_human_query = ""
        for role, text in reversed(history):
            if role == "human":
                last_human_query = text
                break
                
        return f"{last_human_query} {inputs['input']}"

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=get_contextualized_question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    chat_history = []

    print("\nAsk a question about the documents (type 'exit' to quit).\n")
    while True:
        try:
            query = input("Your question: ").strip()
        except EOFError:
            break
        
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue
        
        print("Searching and generating answer...")
        try:
            print("\n--- Answer ---")
            full_answer = ""
            for chunk in rag_chain.stream({
                "input": query,
                "chat_history": chat_history
            }):
                print(chunk, end="", flush=True)
                full_answer += chunk
                
            print("\n" + "-" * 60 + "\n")
            
            chat_history.append(("human", query))
            chat_history.append(("ai", full_answer))

        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
