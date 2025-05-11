# RAG-Powered Multi-Agent Q&A System

A powerful document-based question answering system that combines Retrieval-Augmented Generation (RAG) with a multi-agent approach to provide accurate, context-aware answers. This system emphasizes simplicity, reliability and efficient resource usage.



### Core Components

- **Document Processing**: Support for PDF, text, CSV files, and web content
- **Vector Store**: Chroma DB with local persistence for efficient semantic retrieval
- **Embeddings**: Local Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)
- **LLM**: Llama 3 (70B parameter model via Groq API)
- **Multi-Agent System**: Structured chat agent with specialized tools
- **UI**: Streamlit interface for document upload and querying
- **Tracing**: Optional LangChain tracing for debugging and analysis

### Tools

- **RAG Pipeline**: Primary tool for document-based Q&A
- **Calculator**: For mathematical operations and calculations
- **Dictionary**: Web search powered by Tavily for definitions and explanations

## Key Design Choices

1. **Local Embeddings**: Uses HuggingFace's sentence-transformers locally to avoid API costs and latency
2. **Structured Agent**: Leverages LangChain's structured chat agent for better tool selection logic
3. **Tool Routing**: Simple keyword-based routing for common query patterns with fallback to agent-based decisions
4. **Separate Ingestion Script**: Dedicated script for batch document processing separate from the UI
5. **Contextual Transparency**: Exposes retrieved context chunks for user verification
6. **Environmental Configuration**: Uses dotenv for API key management
7. **Automatic Cleanup**: Temporary files created during document processing are automatically removed
8. **Fault Tolerance**: Graceful handling of missing API keys and embedding model failures

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com/)
- [Tavily API key](https://tavily.com/) (optional for dictionary tool)
- [LangChain API key](https://smith.langchain.com/) (optional for tracing)

### Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd rag-multi-agent-qa
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys
   ```
   GROQ_API_KEY="your-groq-api-key"
   TAVILY_API_KEY="your-tavily-api-key"  # Optional
   LANGCHAIN_API_KEY="your-langchain-api-key"  # Optional
   ```

## Usage

### Option 1: Document Ingestion via Script

Process documents in batch mode:

```bash
python ingest_documents.py --files document1.pdf document2.csv notes.txt --urls "https://example.com/page1" "https://example.com/page2"
```

### Option 2: Run the Streamlit App

```bash
streamlit run app.py
```

Then:
1. Upload documents or enter URLs in the sidebar
2. Click "Process Docs" to ingest and index content
3. Ask questions in the main input field
4. View answers and relevant context

## Example Queries

- "What topics are covered in the syllabus?" (RAG)
- "Calculate 24 * 7 + 365" (Calculator)
- "Define artificial intelligence" (Dictionary)

## Query Routing Logic

Queries are routed to the appropriate tool based on:
- Queries starting with "calculate" go to the Calculator tool
- Queries starting with "define" go to the Dictionary tool
- All other queries use the RAG pipeline with agent deciding which tool is best

## Optimization Opportunities

While the current implementation is functional, there are several areas for potential improvement:

1. **Improved Error Handling**: Enhance exception handling around file operations and API calls
2. **Code Deduplication**: Extract common functionality between `app.py` and `ingest_documents.py`
3. **Robust Temporary File Management**: Use `tempfile` module for more secure temporary file handling
4. **Session Management**: Better cleanup of resources when Streamlit session resets
5. **Async Processing**: Implement async operations for improved UI responsiveness
6. **Progress Indicators**: Add more detailed progress indicators during document processing
7. **Caching**: Implement more aggressive caching for expensive operations
8. **Embedding Model Fallbacks**: Add fallback options if primary embedding model fails

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.