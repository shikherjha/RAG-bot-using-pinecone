import os
# Set environment variables to disable Streamlit's file watching
os.environ["STREAMLIT_WATCH_SERVICE"] = "none"
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["USER_AGENT"] = "rag-multi-agent-qa/1.0 (+you@you.com)"

# Try to handle PyTorch path issues if pysqlite3 is available
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    import sys
    # Continue without the SQLite patch

import re
import math
import streamlit as st
# Display version info to help debug deployment issues
st.sidebar.markdown("### Debug Info")
with st.sidebar.expander("Python and Package Versions"):
    st.code(f"Python version: {sys.version}")
    st.code(f"Sys path: {sys.path[:2]}")  # Show first two entries

# Handle secrets first - simplified approach for Streamlit Cloud
try:
    # Direct access to Streamlit secrets
    if hasattr(st, 'secrets') and st.secrets:
        for key in st.secrets:
            if isinstance(st.secrets[key], str):  # Only set string values
                os.environ[key] = st.secrets[key]
        st.sidebar.success("✅ Loaded Streamlit secrets")
except Exception as e:
    st.sidebar.warning(f"⚠️ Error loading secrets: {str(e)}")

# Fallback to .env if not in cloud and dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    st.sidebar.info("📄 Attempted to load .env file")
except ImportError:
    st.sidebar.info("💡 dotenv not available, using system environment")

# Show which packages are installed - safer implementation
try:
    installed_packages = {}
    import pkg_resources
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    with st.sidebar.expander("Installed Packages"):
        for pkg in ["langchain", "langchain-community", "langchain-core", "streamlit", 
                   "sentence-transformers", "pinecone-client", "langchain-pinecone", "torch", "langchain-groq", 
                   "langchain-huggingface"]:
            version = installed_packages.get(pkg, "Not installed")
            st.code(f"{pkg}: {version}")
except Exception as e:
    st.sidebar.warning(f"Could not list packages: {e}")

# Set up LangChain tracing if available
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-multi-agent-qa")

# App header
st.title("RAG-Powered Multi-Agent Q&A")
st.write("Upload documents or provide URLs to ask questions about their content.")

# Import dependencies with clear error handling
dependencies_ok = True

# 1. LLM setup
llm = None
try:
    from langchain_groq import ChatGroq
    
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-70b-8192",
            temperature=0.2,
            max_tokens=1024,
        )
        st.sidebar.success("✅ LLM initialized")
    else:
        st.sidebar.error("❌ Missing GROQ_API_KEY")
        dependencies_ok = False
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import ChatGroq: {e}")
    dependencies_ok = False
    st.sidebar.info("📦 Try running: pip install langchain-groq")

# 2. Embeddings setup
embeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    
    @st.cache_resource(show_spinner="Loading embedding model...")
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    embeddings = get_embeddings()
    st.sidebar.success("✅ Embeddings initialized")
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import embeddings: {e}")
    dependencies_ok = False
    st.sidebar.info("📦 Try running: pip install langchain-huggingface sentence-transformers")
except Exception as e:
    st.sidebar.error(f"❌ Error loading embeddings: {e}")
    dependencies_ok = False

# 3. Document loaders
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
    from langchain_community.document_loaders.csv_loader import CSVLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.sidebar.success("✅ Document loaders ready")
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import document loaders: {e}")
    dependencies_ok = False
    st.sidebar.info("📦 Try running: pip install langchain-community pypdf")

# 4. Vector store - Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    
    # Verify Pinecone credentials
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENV")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if pinecone_api_key and pinecone_environment and pinecone_index_name:
        st.sidebar.success("✅ Pinecone vector store ready")
    else:
        missing = []
        if not pinecone_api_key: missing.append("PINECONE_API_KEY")
        if not pinecone_environment: missing.append("PINECONE_ENV")
        if not pinecone_index_name: missing.append("PINECONE_INDEX_NAME")
        st.sidebar.error(f"❌ Missing Pinecone credentials: {', '.join(missing)}")
        dependencies_ok = False
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import Pinecone: {e}")
    dependencies_ok = False
    st.sidebar.info("📦 Try running: pip install pinecone-client langchain-pinecone")

# 5. Core components and tools
try:
    from langchain_core.tools import Tool
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    # Simplified tool initialization - no Tavily yet
    def calculator_tool(query: str) -> str:
        safe = re.sub(r"[^0-9+\-*/().^\s]", "", query).replace("^", "**")
        try:
            result = eval(safe, {"__builtins__": {}}, {"math": math})
            return f"Result of {query} = {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    st.sidebar.success("✅ Core components ready")
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import core components: {e}")
    dependencies_ok = False
    st.sidebar.info("📦 Try running: pip install langchain-core")

# 6. Agent setup
try:
    from langchain import hub
    from langchain.agents import create_structured_chat_agent, AgentExecutor
    
    tavily_available = False
    try:
        from langchain_community.tools.tavily_search.tool import TavilySearchResults
        if os.getenv("TAVILY_API_KEY"):
            tavily_available = True
            st.sidebar.success("✅ Tavily search available")
        else:
            st.sidebar.warning("⚠️ Tavily API key missing")
    except ImportError:
        st.sidebar.warning("⚠️ Tavily module not available")
    
    st.sidebar.success("✅ Agent components ready")
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import agent components: {e}")
    dependencies_ok = False
    st.sidebar.info("📦 Try running: pip install langchain tavily-python")

# Main functionality
if not dependencies_ok:
    st.error("Some dependencies failed to load. Please check the sidebar for details.")
    st.stop()

# Helper functions
def load_and_process_documents(file_paths=None, urls=None):
    docs = []
    for path in file_paths or []:
        try:
            loader = PyPDFLoader(path) if path.lower().endswith(".pdf") \
                    else CSVLoader(path) if path.lower().endswith(".csv") \
                    else TextLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
    
    for url in urls or []:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    if not docs:
        return []
    return text_splitter.split_documents(docs)

def initialize_pinecone():
    """Initialize Pinecone client and ensure the index exists"""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            # Create the index if it doesn't exist
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension is 384
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV"))
            )
            st.sidebar.info(f"Created new Pinecone index: {index_name}")
        
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        return None

def setup_vectorstore(chunks):
    if not chunks:
        return None
        
    try:
        # Initialize Pinecone index
        _ = initialize_pinecone()
        
        # Create vector store
        return PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            namespace="rag-documents"  # Optional namespace for organization
        )
    except Exception as e:
        st.error(f"Error creating Pinecone vector store: {e}")
        return None

def setup_retrieval_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    template = """Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def setup_agent(rag_chain):
    tools = [
        Tool(name="Calculator", func=calculator_tool, description="Perform math calculations")
    ]
    
    # Add Tavily if available
    if tavily_available:
        tools.append(Tool(
            name="Dictionary",
            func=TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=1),
            description="Define terms or search for information"
        ))
    
    # Add RAG tool
    tools.append(Tool(name="RAG", func=rag_chain.invoke, description="Answer questions from documents"))
    
    # Pull the prompt template
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Document upload/URL ingestion
uploaded = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
urls = st.sidebar.text_area("Or enter URLs (one per line)").splitlines()

if st.sidebar.button("Process Docs"):
    with st.spinner("Processing documents..."):
        paths, vs_urls = [], [u for u in urls if u.strip()]
        for f in uploaded:
            p = f"./temp_{f.name}"
            with open(p, "wb") as out: 
                out.write(f.getbuffer())
            paths.append(p)
        
        chunks = load_and_process_documents(paths, vs_urls)
        if chunks:
            vs = setup_vectorstore(chunks)
            if vs:
                st.session_state.vs = vs
                st.sidebar.success(f"✅ Indexed {len(chunks)} chunks in Pinecone")
            else:
                st.error("Failed to create vector store")
        else:
            st.warning("No content found to process")
        
        # Clean up temp files
        for p in paths:
            try: os.remove(p)
            except: pass

# Q&A interface
query = st.text_input("Ask a question")
if query and "vs" in st.session_state:
    with st.spinner("Thinking..."):
        vs = st.session_state.vs
        rag_chain = setup_retrieval_chain(vs)
        agent_exec = setup_agent(rag_chain)

        # Tool routing
        if query.lower().startswith("calculate"):
            st.markdown("### Calculator")
            answer = calculator_tool(query)
        elif query.lower().startswith("define") and tavily_available:
            st.markdown("### Dictionary")
            res = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=1)(query)
            answer = res[0]["content"] if res else "No definition found."
        else:
            st.markdown("### RAG Pipeline")
            docs = vs.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
            with st.expander("Retrieved Context"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(d.page_content[:300] + ("…" if len(d.page_content) > 300 else ""))
            
            result = agent_exec.invoke({"input": query})
            answer = result.get("output", "No answer returned.")

        st.markdown("### Answer")
        st.write(answer)
elif query:
    st.warning("Please process documents first")
else:
    st.info("Upload documents and ask questions to get started!")