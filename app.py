import os
# Disable Streamlit’s file watcher errors and identify your requests
os.environ["STREAMLIT_WATCH_SERVICE"] = "none"
os.environ["USER_AGENT"] = "rag-multi-agent-qa/1.0 (+jhashikher@gmail.com)"

import re
import math
import streamlit as st
from dotenv import load_dotenv

# 1) Load .env for any API keys (Groq, Tavily, LangChain tracing)
load_dotenv()

# Fallback to Streamlit secrets if running in Cloud
from streamlit import secrets
for _key, _val in secrets.items():
    os.environ.setdefault(_key, _val)

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-multi-agent-qa")

# 2) LLM setup (Groq)
from langchain_groq import ChatGroq
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0.2,
    max_tokens=1024,
)

# 3) Local embeddings (no API keys)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Embedding load failed: {e}")
        return None

embeddings = get_embeddings()

# 4) Document loading & splitting
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_process_documents(file_paths=None, urls=None):
    docs = []
    for path in file_paths or []:
        loader = PyPDFLoader(path) if path.lower().endswith(".pdf") \
                 else CSVLoader(path) if path.lower().endswith(".csv") \
                 else TextLoader(path)
        docs.extend(loader.load())
    for url in urls or []:
        docs.extend(WebBaseLoader(url).load())
    return text_splitter.split_documents(docs)

# 5) Vector store creation
from langchain_community.vectorstores import Chroma

def setup_vectorstore(chunks):
    if embeddings is None:
        st.error("No embeddings available.")
        return None
    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None

# 6) Tools
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults

def calculator_tool(query: str) -> str:
    safe = re.sub(r"[^0-9+\-*/().^\s]", "", query).replace("^", "**")
    try:
        result = eval(safe, {"__builtins__": {}}, {"math": math})
        return f"Result of {query} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# 7) RAG chain helper
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

# 8) Agent setup with default prompt from Hub
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.callbacks import LangChainTracer

def setup_agent(rag_chain):
    tools = [
        Tool("Calculator", calculator_tool, "Perform math"),
        Tool(
            "Dictionary",
            TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=1),
            "Define terms"
        ),
        Tool("RAG", rag_chain.invoke, "Answer from docs")
    ]
    # Pull the built-in structured-chat prompt (includes tool_names/tools/agent_scratchpad)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt)
    tracer = LangChainTracer() if os.getenv("LANGCHAIN_API_KEY") else None
    return AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=[tracer] if tracer else [])

# 9) Streamlit UI
def main():
    st.title("RAG-Powered Multi-Agent Q&A")

    # Sidebar: API status
    with st.sidebar.expander("API Status"):
        st.write(f"Groq: {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")
        st.write(f"Tavily: {'✅' if os.getenv('TAVILY_API_KEY') else '❌'}")
        if embeddings is None:
            st.error("Embeddings failed to load.")

    # Document upload/URL ingestion
    uploaded = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    urls = st.sidebar.text_area("Or enter URLs (one per line)").splitlines()
    if st.sidebar.button("Process Docs"):
        paths, vs_urls = [], [u for u in urls if u.strip()]
        for f in uploaded:
            p = f"./temp_{f.name}"
            with open(p, "wb") as out: out.write(f.getbuffer())
            paths.append(p)
        chunks = load_and_process_documents(paths, vs_urls)
        vs = setup_vectorstore(chunks)
        if vs:
            st.session_state.vs = vs
            st.sidebar.success(f"Indexed {len(chunks)} chunks")
        for p in paths:
            try: os.remove(p)
            except: pass

    # Q&A interface
    query = st.text_input("Ask a question")
    if query and "vs" in st.session_state:
        vs = st.session_state.vs
        rag_chain = setup_retrieval_chain(vs)
        agent_exec = setup_agent(rag_chain)

        # Tool routing: simple keyword check
        if query.lower().startswith("calculate"):
            st.markdown("### Calculator")
            answer = calculator_tool(query)
        elif query.lower().startswith("define"):
            st.markdown("### Dictionary")
            if os.getenv("TAVILY_API_KEY"):
                res = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"), max_results=1)(query)
                answer = res[0]["content"] if res else "No definition found."
            else:
                answer = "Dictionary unavailable."
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
        st.warning("Please process documents first.")

if __name__ == "__main__":
    main()
