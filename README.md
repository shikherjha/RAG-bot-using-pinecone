# RAG-Powered Multi-Agent Q\&A System

A powerful document-based question answering system that combines Retrieval-Augmented Generation (RAG) with a multi-agent approach to provide accurate, context-aware answers. This system emphasizes simplicity, reliability, and efficient resource usage.

---

## Core Components

* **Document Processing**: Supports PDF, text, CSV files, and web content.
* **Vector Store**: Pinecone vector database (Serverless, dimension=1024, cosine similarity).
* **Embeddings**: Local Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
* **LLM**: Llama 3 (70B parameters via Groq API).
* **Multi-Agent System**: Structured chat agent with specialized tools.
* **UI**: Streamlit interface for document upload and querying.
* **Tracing**: Optional LangChain tracing for debugging and analysis.

## Tools

* **RAG Pipeline**: Primary tool for document-based Q\&A (via Pinecone retrieval).
* **Calculator**: For mathematical operations and calculations.
* **Dictionary**: Web search powered by Tavily for definitions and explanations.

## Key Design Choices

1. **Pinecone Vector Store**: Managed, serverless index for scalable semantic search.
2. **Local Embeddings**: Uses HuggingFace's sentence-transformers locally to avoid external embedding API costs and reduce latency.
3. **Structured Agent**: Leverages LangChain's structured chat agent for intelligent tool selection.
4. **Tool Routing**: Keyword-based routing for common queries with fallback to agent-based selection.
5. **Separate Ingestion Script**: Dedicated `ingest_documents.py` for batch document processing.
6. **Context Transparency**: Exposes retrieved context chunks for user verification.
7. **Environmental Configuration**: Uses `.env` or Streamlit secrets for API key management (Groq, Pinecone, Tavily, LangChain).
8. **Automatic Cleanup**: Temporary files removed automatically after processing.
9. **Fault Tolerance**: Graceful handling of missing API keys and embedding/model failures.

## Setup Instructions

### Prerequisites

* Python 3.10+
* [Groq API key](https://console.groq.com/)
* [Pinecone account](https://app.pinecone.io/) (Starter plan for free tier)
* [Tavily API key](https://tavily.com/) (optional for Dictionary tool)
* [LangChain API key](https://smith.langchain.com/) (optional for tracing)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd rag-multi-agent-qa
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   Create a `.env` file (or use Streamlit Secrets) with:

   ```ini
   GROQ_API_KEY="your-groq-api-key"
   PINECONE_API_KEY="your-pinecone-api-key"
   PINECONE_ENV="us-east-1-aws"          # replace with your Pinecone environment
   PINECONE_INDEX_NAME="your-index-name"
   TAVILY_API_KEY="your-tavily-api-key"  # optional
   LANGCHAIN_API_KEY="your-langchain-api-key"  # optional
   ```
4. **Create Pinecone Index**

   * Via Console: Set dimension=1024, metric=cosine, capacity mode=Serverless, region=us-east-1.
   * Or programmatically, see `ingest_documents.py` startup logic for auto-creation.

## Usage

### 1. Document Ingestion (Batch)

Process documents and URLs in bulk:

```bash
python ingest_documents.py --files document1.pdf document2.csv notes.txt --urls "https://example.com/page1" "https://example.com/page2"
```

This script will create or verify the Pinecone index, compute embeddings, and upsert vectors.

### 2. Streamlit App

Run the interactive UI:

```bash
streamlit run app.py
```

Then:

1. Upload documents or enter URLs in the sidebar.
2. Click **"Process Docs"** to ingest and index content.
3. Ask questions in the main input field.
4. View answers and relevant context.

## Example Queries

* **RAG**: "What topics are covered in the syllabus?"
* **Calculator**: "Calculate 24 \* 7 + 365"
* **Dictionary**: "Define artificial intelligence"

## Query Routing Logic

* Queries starting with **"calculate"** use the Calculator tool.
* Queries starting with **"define"** use the Dictionary (Tavily) tool.
* All others use the RAG pipeline with Pinecone retrieval and agent reasoning.

## Future Improvements

* **Error Handling**: More robust exception handling around file, API, and network operations.
* **Code Refactoring**: Extract shared logic between `app.py` and `ingest_documents.py` into common modules.
* **Temp File Management**: Use Python’s `tempfile` for safer temporary file operations.
* **Async UI**: Asynchronous document processing for better responsiveness.
* **Progress Bars**: Detailed progress feedback during ingestion.
* **Caching Strategies**: Advanced caching for embeddings and retrieval.
* **Fallback Embeddings**: Alternative embedding models if the primary model fails.

## Contributing

Contributions are welcome! Please submit a Pull Request or open an issue for suggestions.

## License

This project is licensed under the MIT License.
