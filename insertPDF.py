import os
import threading
import chromadb
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
collection_name = "books_library"

# Ensure ChromaDB is clean before inserting (DELETE OLD DATA)
try:
    chroma_client.delete_collection(collection_name)
    print(f"Deleted old collection '{collection_name}' to avoid embedding conflicts.")
except Exception:
    pass  # Ignore errors if collection didn't exist

# Create a new collection
collection = chroma_client.get_or_create_collection(name=collection_name)

# Define root folder where PDFs are stored
PDF_ROOT_FOLDER = "Libros/"

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize embedding model (Ensures 768-dim embeddings)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Thread Lock for ChromaDB
lock = threading.Lock()
BATCH_SIZE = 100  #  Batch size for efficient insertions

def process_pdfs():
    """ Process all PDFs using a thread pool and insert chunks immediately. """
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for topic_folder in sorted(os.listdir(PDF_ROOT_FOLDER)):  
            topic_path = os.path.join(PDF_ROOT_FOLDER, topic_folder)

            if os.path.isdir(topic_path):
                pdf_files = sorted([f for f in os.listdir(topic_path) if f.endswith(".pdf")])
                if not pdf_files:
                    print(f"⚠️ No PDFs found in {topic_folder}, skipping...")
                    continue  

                print(f"\n Processing Topic: {topic_folder} ({len(pdf_files)} PDFs)...")

                for pdf_file in pdf_files:
                    file_path = os.path.join(topic_path, pdf_file)
                    futures[executor.submit(process_single_pdf, file_path, topic_folder)] = pdf_file

        # Wait for all PDFs to finish processing
        for future in as_completed(futures):
            pdf_name = futures[future]
            try:
                total_inserted = future.result()
                print(f"[{pdf_name}] Inserted {total_inserted} items into ChromaDB.")
                print(f"Finished processing: {pdf_name}")
            except Exception as e:
                print(f"Error processing {pdf_name}: {e}")

    print("All PDFs Processed Successfully!")

def process_single_pdf(pdf_path, topic):
    """ Load and split a PDF into text chunks, then insert directly into ChromaDB. """
    
    pdf_name = os.path.basename(pdf_path)
    print(f"Processing: {pdf_name}...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for i, page in enumerate(pages):
        page.metadata = {
            "topic": topic,
            "filename": pdf_name,
            "page_number": i + 1  
        }

    split_docs = text_splitter.split_documents(pages)
    return insert_documents_in_batches(split_docs, pdf_name)

def insert_documents_in_batches(documents, pdf_name):
    """ Inserts document chunks into ChromaDB in parallel batches. """
    if not documents:
        return 0

    inserted_count = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            futures.append(executor.submit(flush_batch, batch, pdf_name))

        for future in as_completed(futures):
            inserted_count += future.result()

    return inserted_count

def flush_batch(batch, pdf_name):
    """ Efficiently insert a batch of documents into ChromaDB. """
    if not batch:
        return 0

    try:
        ids = [
            f"{doc.metadata['filename']}_p{doc.metadata['page_number']}_{hashlib.md5(doc.metadata['filename'].encode()).hexdigest()}_{uuid.uuid4().hex[:8]}"
            for doc in batch
        ]
        embeddings = [embedding_model.embed_query(doc.page_content) for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        documents = [doc.page_content for doc in batch]

        with lock:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

        return len(batch)

    except Exception as e:
        print(f"[{pdf_name}] Failed to insert batch: {str(e)}")
        return 0

print("Starting Optimized PDF Processing...")
process_pdfs()
print("\n All books inserted! Exiting...\n")
os._exit(0)
