import os
import asyncio
import psutil
import chromadb
import re
from dotenv import load_dotenv
from collections import deque
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings  # Ensure same embeddings

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

# Initialize embedding model (Must match insertion)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Azure OpenAI for response generation
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.7,  # More natural variation in responses
    streaming=True
)

def measure_system_metrics():
    """Measure memory and CPU usage."""
    memory_usage = psutil.virtual_memory().used / (1024 * 1024)
    cpu_utilization = psutil.cpu_percent()
    return memory_usage, cpu_utilization

def clean_retrieved_text(text):
    """Cleans unwanted content from retrieved documents."""
    # Remove unnecessary table of contents, index, metadata, and other noisy sections
    text = re.sub(r"\b(Table of Contents|Index|References|Chapter \d+)\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()  # Remove excessive whitespace
    return text

def retrieve_relevant_info(query, top_k=5):
    """Retrieve, clean, and summarize relevant documents from ChromaDB."""
    try:
        collection = chroma_client.get_collection("books_library")
    except Exception as e:
        return None, f"⚠️ Error accessing ChromaDB: {e}"

    sources_summary = {}
    relevant_chunks = []

    try:
        # Use correct embeddings for search
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if documents and metadatas:
            for doc_list, meta_list in zip(documents, metadatas):
                for doc, meta in zip(doc_list, meta_list):
                    pdf_source = meta.get("filename", "Unknown PDF")
                    page_number = meta.get("page_number", "Unknown Page")

                    # Store pages for source tracking
                    if pdf_source not in sources_summary:
                        sources_summary[pdf_source] = set()
                    sources_summary[pdf_source].add(page_number)

                    # Clean extracted text before using
                    cleaned_text = clean_retrieved_text(doc)
                    relevant_chunks.append(cleaned_text)

    except Exception as e:
        return None, f"Error querying database: {e}"

    # Prepare source reference string (e.g., "Book.pdf (Pages 1, 3, 5)")
    sources_text = "\n\n**Sources Used:**\n" + "\n".join(
        [f"*{pdf}* (Pages {', '.join(map(str, sorted(pages)))})" for pdf, pages in sources_summary.items()]
    ) if sources_summary else ""

    # Return cleaned text chunks and formatted source references
    return "\n\n".join(relevant_chunks) if relevant_chunks else None, sources_text

chat_history = deque(maxlen=10)

async def stream_chat_response(ai_response):
    """Streams AI responses in real time."""
    for chunk in ai_response.split():
        print(chunk, end=" ", flush=True)
        await asyncio.sleep(0.02)
    print("\n")

async def main():
    """Chatbot using ChromaDB for retrieval and Azure OpenAI for response generation."""
    print("Welcome to the AI Chatbot with ChromaDB! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        relevant_text, sources_text = retrieve_relevant_info(user_input)

        # If no relevant info is found, return a message instead of making something up
        if not relevant_text:
            print("\nAI: No relevant information found in the available documents.")
            continue

        # Memory & CPU Usage Display
        memory, cpu = measure_system_metrics()
        print(f"\nSystem Metrics: Memory Usage: {memory:.2f} MB | CPU Utilization: {cpu:.2f}%\n")

        messages = [
            SystemMessage(content="You are an AI assistant that answers only based on PDF documents stored in ChromaDB. Do not generate information beyond the given sources."),
            HumanMessage(content=f"User asked: {user_input}\n\nHere is the relevant information:\n{relevant_text}")
        ]

        try:
            ai_response = await llm.ainvoke(messages)
            chat_history.append((user_input, ai_response.content))  

            print("\nAI: ", end="")
            await stream_chat_response(ai_response.content)

            # Display clean source references after AI response
            print(sources_text)

        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    asyncio.run(main())
