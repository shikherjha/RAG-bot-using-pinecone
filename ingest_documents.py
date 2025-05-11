import os
import argparse
from dotenv import load_dotenv

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

# 4) Vector store
from langchain_community.vectorstores import Chroma

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database")
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

    # 10) Build or persist vector store
    print("Creating/updating Chroma vector store...")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks into ./chroma_db")
    except Exception as e:
        print("❌ Error creating vector store:", e)

if __name__ == "__main__":
    main()