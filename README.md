# SmartPDFQA - AI Chatbot for PDFs with ChromaDB & Azure OpenAI
SmartPDFQA is an AI-powered chatbot that **reads and understands PDFs**. It allows users to **ask questions**, and **Azure OpenAI answers in a natural way**, using **only** the information stored in the PDFs.

## **How It Works**
**It reads all the PDFs you put in the Libros folder**  
**The system processes them**, extracting text and creating **high-quality embeddings** with Hugging Face
**Embeddings are stored in ChromaDB**, a powerful vector database for fast searches 
**When a user asks a question**, ChromaDB finds the most relevant PDF content  
**Azure OpenAI generates a natural response**, using only the retrieved data 

---

## **Features**
✅ **Process unlimited PDFs** → Just add new files, and they will be indexed.  
✅ **Retrieval-Augmented Generation (RAG)** → Ensures answers are **only based on PDFs**.  
✅ **Fast and efficient searches** → Uses **ChromaDB** for retrieval.  
✅ **Accurate embeddings** → Uses `sentence-transformers/all-mpnet-base-v2` for high-quality text understanding.  
✅ **Natural language responses** → **Azure OpenAI** rephrases information for readability.  
✅ **Provides sources & page numbers** → So users can verify the information.  

---

Example of using the chat

Welcome to the AI Chatbot with ChromaDB! Type 'exit' to quit.

You: What are the main components of the Android operating system?

AI: The main components of the Android operating system include:
1. **Linux Kernel**: Provides the core system functionalities.
2. **Android Runtime (ART)**: Manages memory and executes code.
3. **Application Framework**: Provides API access for developers.
4. **System Apps**: Built-in applications like settings and phone dialer.
5. **Hardware Abstraction Layer (HAL)**: Interfaces between Android OS and device hardware.

📄 *Android Software Internals Quick Reference.pdf* (Pages 7, 45, 78, 102)



## 🔹 Environment Variables

Create a `.env` file in the project root and add the following credentials:

```ini
# Azure OpenAI Credentials
OPENAI_API_KEY=your-azure-openai-key
OPENAI_API_VERSION=your-api-version
AZURE_ENDPOINT=your-azure-endpoint
DEPLOYMENT_NAME=your-azure-deployment
MODEL_NAME=gpt-4  # or your preferred model

# ChromaDB Credentials
CHROMADB_HOST=chromadb_env
CHROMADB_PORT=8000
CHROMADB_USER=admin
CHROMADB_PASSWORD=securepassword123


Run the docker-compose.yml file with docker-compose up --build -d
This will setup docker images with ChromaDB and python development environment
