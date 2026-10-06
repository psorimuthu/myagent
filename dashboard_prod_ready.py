import os
import logging
import tempfile
from typing import List, Dict, Any
from typing_extensions import TypedDict
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# 1. Page Configuration & Layout
st.set_page_config(page_title="Production Agentic RAG", layout="wide")
st.title("🤖 Cloud Agentic RAG (LangGraph + Pinecone)")
st.markdown("This workspace verifies internal documents. If data is missing, it **halts** and awaits your approval before hitting the live web.")

# Wire secure cloud variables from Streamlit Advanced Secrets Settings
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

INDEX_NAME = "my-rag-index" 

# 2. Initialize Infrastructure and Cached Components
@st.cache_resource
def setup_infrastructure():
    # Force output dimensions to 1024 to align with your Pinecone cluster setting
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=1024)
    db = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    search_tool = DuckDuckGoSearchRun()
    return db, llm, search_tool

db, llm, search_tool = setup_infrastructure()
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. Define LangGraph State and Cognitive Layout Nodes
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    search_needed: bool

@st.cache_resource
def build_agent_graph():
    def retrieve_node(state: AgentState):
        matched_docs = retriever.invoke(state["question"])
        return {"documents": [d.page_content for d in matched_docs], "question": state["question"]}

    def grade_documents_node(state: AgentState):
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict data grader. Reply with 'YES' if relevant or 'NO' if it is not."),
            ("human", f"Context: {state['documents']}\n\nQuestion: {state['question']}")
        ])
        assessment = (grader_prompt | llm | StrOutputParser()).invoke({}).strip().upper()
        return {"search_needed": "YES" not in assessment}

    def web_search_node(state: AgentState):
        print("--- NODE: EXECUTING LIVE WEB SEARCH ---")
        try:
            web_results = search_tool.invoke({"query": state["question"]})
            return {"documents": [web_results]}
        except Exception:
            return {"documents": ["SYSTEM NOTICE: Cloud environment connection timeout during web fallback lookups."]}

    def generate_answer_node(state: AgentState):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question comprehensively using only the context blocks below:\n\n{context}"),
            ("human", "{question}")
        ])
        answer = (qa_prompt | llm | StrOutputParser()).invoke({"context": "\n\n".join(state["documents"]), "question": state["question"]})
        return {"generation": answer}

    def decide_next_step(state: AgentState) -> str:
        if state.get("search_needed", True):
            return "web_search"
        else:
            return "generate"

    # Assemble Structural Pipeline Workflow Flowlines
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    workflow.add_conditional_edges("grade_docs", decide_next_step, {"web_search": "web_search", "generate": "generate"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["web_search"])

agent_graph = build_agent_graph()

# 4. Streamlit Persistent Session States Management Setup
if "ui_chat_history" not in st.session_state:
    st.session_state.ui_chat_history = []
if "graph_config" not in st.session_state:
    st.session_state.graph_config = {"configurable": {"thread_id": "streamlit_session_101"}}
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False
if "visual_docs" not in st.session_state:
    st.session_state.visual_docs = [] # Local dual-storage memory cache backup array for visual maps

# =====================================================================
# SIDEBAR DYNAMIC DOCUMENT INGESTION
# =====================================================================
with st.sidebar:
    st.info("☁️ Connected to Cloud-Native Pinecone Index Serverless Farms.")
    uploaded_file = st.file_uploader("Upload a new PDF to your cloud repository:", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Index Document", use_container_width=True):
            with st.spinner("Streaming vector fragments up to Pinecone Cloud infrastructure..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    loader = PyPDFLoader(tmp_file_path)
                    raw_docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = text_splitter.split_documents(raw_docs)
                    
                    for d in split_docs:
                        d.metadata["source"] = uploaded_file.name
                    
                    # Task A: Upload raw vectors to Pinecone indefinitely
                    db.add_documents(split_docs)
                    
                    # Task B: Convert text to embeddings locally and add to sync state for the 3D Graph
                    raw_embeddings = db.embeddings.embed_documents([d.page_content for d in split_docs])
                    for i, doc in enumerate(split_docs):
                        st.session_state.visual_docs.append({
                            "text": doc.page_content,
                            "embedding": raw_embeddings[i]
                        })
                        
                    st.success(f"Successfully processed and indexed cloud parameters!")
                except Exception as e:
                    st.error(f"Ingestion lifecycle failed: {e}")
                finally:
                    os.unlink(tmp_file_path)
                    st.rerun()

# =====================================================================
# LAYOUT DEFINITION: SPLIT WORKSPACE
# =====================================================================
chat_col, visual_col = st.columns(2)

# --- LEFT COLUMN: CONTROL INTERFACE ---
with chat_col:
    st.subheader("Interactive Agent Interface")

    # Render persistent conversation chat bubbles
    for role, text in st.session_state.ui_chat_history:
        with st.chat_message(role):
            st.write(text)

    # Human-in-the-Loop Form block (Shows up ONLY during a LangGraph block interruption)
    if st.session_state.awaiting_approval:
        st.warning("⚠️ **Agent Interrupted:** Information missing from database. Web search requested.")
        with st.form("hitl_form"):
            override_query = st.text_input("Modify the web search parameters (leave blank to run original):", "")
            f_col1, f_col2 = st.columns(2)
            approve = f_col1.form_submit_button("✅ Approve Web Search")
            deny = f_col2.form_submit_button("❌ Deny Web Search")
            
            if approve:
                if override_query.strip() != "":
                    agent_graph.update_state(st.session_state.graph_config, {"question": override_query.strip(), "search_needed": True}, as_node="grade_docs")
                    st.session_state.ui_chat_history.append(("assistant", f"✍️ Supervisor updated query to: '{override_query.strip()}'"))
                
                with st.spinner("Executing live web search..."):
                    for event in agent_graph.stream(None, st.session_state.graph_config, stream_mode="values"):
                        pass
                
                final_state = agent_graph.get_state(st.session_state.graph_config)
                st.session_state.ui_chat_history.append(("assistant", final_state.values.get("generation", "Error compiling response.")))
                st.session_state.awaiting_approval = False
                st.rerun()
                
            if deny:
                agent_graph.update_state(st.session_state.graph_config, {"generation": "Web search denied by human supervisor.", "search_needed": False}, as_node="grade_docs")
                st.session_state.ui_chat_history.append(("assistant", "❌ Web search cancelled by human supervisor."))
                st.session_state.awaiting_approval = False
                st.rerun()

    # Normal user chat element input container
    if not st.session_state.awaiting_approval:
        if user_input := st.chat_input("Ask a question about your documents..."):
