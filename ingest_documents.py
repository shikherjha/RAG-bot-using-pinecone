import os
import argparse
from dotenv import load_dotenv

# Try to handle SQLite issues if pysqlite3 is available
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Continue without the SQLite patch

# 0) User-Agent to avoid anonymous WebBaseLoader requests
os.environ["USER_AGENT"] = "rag-ingest-script/1.0 (+jhashikher@gmail.com)"

# 1) Load environment variables from .env
load_dotenv()

# Fallback to Streamlit secrets if running in Cloud (optional)
try:
    from streamlit import secrets
    for _key, _val in secrets.items():
        os.environ.setdefault(_key, _val)
except ImportError:
    pass

# Optional: LangChain tracing if you're using LangSmith
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-multi-agent-qa")

# 2) Imports for loading and splitting - SIMPLIFIED
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 3) Local embeddings (no API key)
from langchain_huggingface import HuggingFaceEmbeddings

# 4) Vector store - Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

def initialize_pinecone():
    """Initialize Pinecone and ensure index exists"""
    try:
        # Get Pinecone credentials from environment
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENV")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not all([api_key, environment, index_name]):
            print("❌ Missing Pinecone credentials. Check your .env file.")
            missing = []
            if not api_key: missing.append("PINECONE_API_KEY")
            if not environment: missing.append("PINECONE_ENV")
            if not index_name: missing.append("PINECONE_INDEX_NAME")
            print(f"   Missing: {', '.join(missing)}")
            return None, None
            
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            # Create the index with proper dimensions for the embedding model
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 uses 384 dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment)
            )
            print(f"✅ Created new Pinecone index: {index_name}")
        else:
            print(f"✅ Using existing Pinecone index: {index_name}")
            
        # Get the index
        index = pc.Index(index_name)
        return pc, index
        
    except Exception as e:
        print(f"❌ Error initializing Pinecone: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Pinecone vector database")
    parser.add_argument("--files", nargs="+", help="Paths to files to ingest")
    parser.add_argument("--urls", nargs="+", help="URLs to ingest")
    args = parser.parse_args()

    # 5) Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Successfully loaded embeddings model")
    except Exception as e:
        print("❌ Failed to load local embeddings:", e)
        print("   Make sure you've installed sentence-transformers:")
        print("   pip install sentence-transformers")
        return

    # 6) Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = []

    # 7) Process files
    if args.files:
        for file_path in args.files:
            print(f"Processing file: {file_path}")
            try:
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.lower().endswith('.csv'):
                    loader = CSVLoader(file_path)
                else:  # plain text
                    loader = TextLoader(file_path)

                docs = loader.load()
                print(f"  → Loaded {len(docs)} pages/documents")
                documents.extend(docs)
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")

    # 8) Process URLs
    if args.urls:
        for url in args.urls:
            print(f"Processing URL: {url}")
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                print(f"  → Loaded {len(docs)} documents from web")
                documents.extend(docs)
            except Exception as e:
                print(f"   Error processing {url}: {e}")

    # 9) Split into chunks
    if not documents:
        print("No documents to process. Provide valid --files or --urls.")
        return

    print(f"Splitting {len(documents)} documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"  → Created {len(chunks)} chunks")

    # 10) Initialize Pinecone
    _, _ = initialize_pinecone()
    
    # 11) Build vector store with Pinecone
    print("Creating/updating Pinecone vector store...")
    try:
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            namespace="rag-documents"  # Optional organization by namespace
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks into Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}")
    except Exception as e:
        print("❌ Error creating vector store:", e)
        print("   Error details:", str(e))

if __name__ == "__main__":
    main()