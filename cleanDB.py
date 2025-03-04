import chromadb
import os

# Load ChromaDB connection
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")

# Connect to ChromaDB
chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=int(CHROMADB_PORT))

# Get all collection names
collection_names = [c for c in chroma_client.list_collections()]

# Delete all collections
for name in collection_names:
    chroma_client.delete_collection(name=name)
    print(f" Deleted collection: {name}")

print(" All data in ChromaDB deleted successfully!")
