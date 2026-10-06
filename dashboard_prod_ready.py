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
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun


# 1. Page Configuration & Layout
st.set_page_config(page_title="Agentic RAG Control Center", layout="wide")
st.title("🤖 Agentic RAG Workspace & Live Vector Tracker")

PERSIST_DIRECTORY = "./chroma_db"

# Secure cloud injection variables
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 2. Define LangGraph State & Core Elements
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    search_needed: bool

@st.cache_resource
def setup_infrastructure():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    # Initialize Chroma to read from our persistent directory
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    search_tool = DuckDuckGoSearchRun()
    return db, llm, search_tool

db, llm, search_tool = setup_infrastructure()
retriever = db.as_retriever(search_kwargs={"k": 2})

@st.cache_resource
def build_agent_graph():
    # Define internal compilation steps
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
        except Exception as network_error:
            error_fallback_text = (
                "SYSTEM NOTICE: An external network connection block occurred. "
                "The assistant was unable to pull live data from DuckDuckGo Search because "
                "the host environment network connection was reset by the peer."
            )
            return {"documents": [error_fallback_text]}

    def generate_answer_node(state: AgentState):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only the context below:\n\n{context}"),
            ("human", "{question}")
        ])
        answer = (qa_prompt | llm | StrOutputParser()).invoke({"context": "\n\n".join(state["documents"]), "question": state["question"]})
        return {"generation": answer}

    def decide_next_step(state: AgentState) -> str:
        if state.get("search_needed", True):
            return "web_search"
        else:
            return "generate"

    # Assemble Pipeline Logic
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

# 3. Streamlit Persistent Session States
if "ui_chat_history" not in st.session_state:
    st.session_state.ui_chat_history = []
if "graph_config" not in st.session_state:
    st.session_state.graph_config = {"configurable": {"thread_id": "streamlit_session_101"}}
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False

# =====================================================================
# SIDEBAR DYNAMIC DOCUMENT UPLOADER & INGESTION
# =====================================================================
with st.sidebar:
    st.subheader("📁 Knowledge Base Ingestion")
    uploaded_file = st.file_uploader("Upload a new PDF to your cloud repository:", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Index Document", use_container_width=True):
            with st.spinner("Slicing and converting document to vector coordinates..."):
                # Save the uploaded uploaded file data stream into a temporary storage path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    # Execute standard loader pipeline extraction
                    loader = PyPDFLoader(tmp_file_path)
                    raw_docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = text_splitter.split_documents(raw_docs)
                    
                    # Force update the custom structural metadata sources tracking key
                    for d in split_docs:
                        d.metadata["source"] = uploaded_file.name
                        
                    # Add documents straight into our global active database instance
                    db.add_documents(split_docs)
                    st.success(f"Successfully indexed {len(split_docs)} text chunks from '{uploaded_file.name}'!")
                except Exception as ingest_error:
                    st.error(f"Ingestion process failed: {ingest_error}")
                finally:
                    os.unlink(tmp_file_path) # Clean up file stream paths safely
                    st.rerun()

# =====================================================================
# LAYOUT RENDERING: SPLIT WORKSPACE
# =====================================================================
chat_col, visual_col = st.columns(2)

# --- LEFT COLUMN: CONTROL INTERFACE ---
with chat_col:
    st.subheader("Interactive Agent Interface")

    for role, text in st.session_state.ui_chat_history:
        with st.chat_message(role):
            st.write(text)

    if st.session_state.awaiting_approval:
        st.warning("⚠️ **Agent Interrupted:** The requested data was not found in internal documents. Web search required.")
        with st.form("hitl_form"):
            override_query = st.text_input("Modify the web search query (leave blank to approve original):", "")
            f_col1, f_col2 = st.columns(2)
            approve = f_col1.form_submit_button("✅ Approve Web Search")
            deny = f_col2.form_submit_button("❌ Deny Web Search")
            
            if approve:
                if override_query.strip() != "":
                    agent_graph.update_state(st.session_state.graph_config, {"question": override_query.strip(), "search_needed": True}, as_node="grade_docs")
                    st.session_state.ui_chat_history.append(("assistant", f"✍️ Supervisor updated query to: '{override_query.strip()}'"))
                with st.spinner("Executing live web search & compiling response..."):
                    for event in agent_graph.stream(None, st.session_state.graph_config, stream_mode="values"):
                        pass
                final_state = agent_graph.get_state(st.session_state.graph_config)
                st.session_state.ui_chat_history.append(("assistant", final_state.values.get("generation", "Error compiling response.")))
                st.session_state.awaiting_approval = False
                st.rerun()
                
            if deny:
                agent_graph.update_state(st.session_state.graph_config, {"generation": "Web search denied by human supervisor.", "search_needed": False}, as_node="grade_docs")
                st.session_state.ui_chat_history.append(("assistant", "❌ Web search denied by human supervisor."))
                st.session_state.awaiting_approval = False
                st.rerun()

    if not st.session_state.awaiting_approval:
        if user_input := st.chat_input("Ask a question..."):
            st.session_state.ui_chat_history.append(("user", user_input))
            with st.spinner("Analyzing document database structures..."):
                for event in agent_graph.stream({"question": user_input}, st.session_state.graph_config, stream_mode="values"):
                    pass
            snapshot = agent_graph.get_state(st.session_state.graph_config)
            if snapshot.next:
                st.session_state.awaiting_approval = True
                st.rerun()
            else:
                st.session_state.ui_chat_history.append(("assistant", snapshot.values.get("generation", "No answer compiled.")))
                st.rerun()

