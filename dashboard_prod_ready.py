import os
import logging
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# 1. Page Configuration & Layout
st.set_page_config(page_title="Agentic RAG Control Center", layout="wide")
st.title("🤖 Agentic RAG Workspace (LangGraph + HITL)")
st.markdown("This assistant verifies internal document context. If info is missing, it **halts** and asks for your approval before searching the web.")

PERSIST_DIRECTORY = "./chroma_db"

if not os.path.exists(PERSIST_DIRECTORY):
    st.error(f"Database at `{PERSIST_DIRECTORY}` not found. Ingest files first!")
    st.stop()

# Initialize API credentials
os.environ["GROQ_API_KEY"] = 
os.environ["GOOGLE_API_KEY"] =

# 2. Define LangGraph State & Infrastructure
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    search_needed: bool

@st.cache_resource
def setup_agent_graph():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
    search_tool = DuckDuckGoSearchRun()

    # Define Graph Nodes
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
        web_results = search_tool.invoke({"query": state["question"]})
        return {"documents": [web_results]}

    def generate_answer_node(state: AgentState):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only the context below:\n\n{context}"),
            ("human", "{question}")
        ])
        answer = (qa_prompt | llm | StrOutputParser()).invoke({"context": "\n\n".join(state["documents"]), "question": state["question"]})
        return {"generation": answer}

    def decide_next_step(state: AgentState) -> str:
        return "web_search" if state["search_needed"] else "generate"

    # Build Graph Pipeline
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
    compiled_app = workflow.compile(checkpointer=memory, interrupt_before=["web_search"])
    return db, compiled_app

db, agent_graph = setup_agent_graph()

# 3. Streamlit Persistent Session States
if "ui_chat_history" not in st.session_state:
    st.session_state.ui_chat_history = []  # Tracks UI visualization list tuples [("user", text), ("assistant", text)]
if "graph_config" not in st.session_state:
    st.session_state.graph_config = {"configurable": {"thread_id": "streamlit_session_101"}}
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False

# =====================================================================
# LAYOUT RENDERING: SPLIT WORKSPACE
# =====================================================================
chat_col, visual_col = st.columns([1, 1])

# --- LEFT COLUMN: CONTROL INTERFACE ---
with chat_col:
    st.subheader("Interactive Agent Interface")

    # Render persistent conversation feed blocks
    for role, text in st.session_state.ui_chat_history:
        with st.chat_message(role):
            st.write(text)

    # Human Intervention Form (Shows up ONLY when the graph triggers a breakpoint)
    if st.session_state.awaiting_approval:
        st.warning("⚠️ **Agent Interrupted:** The requested data was not found in internal documents. Web search required.")
        
        with st.form("hitl_form"):
            override_query = st.text_input("Modify the web search query (leave blank to approve original):", "")
            
            f_col1, f_col2 = st.columns(2)
            approve = f_col1.form_submit_button("✅ Approve Web Search")
            deny = f_col2.form_submit_button("❌ Deny Web Search")
            
            if approve:
                if override_query.strip() != "":
                    # Inject rewritten text state into the thread checkpointer memory
                    agent_graph.update_state(st.session_state.graph_config, {"question": override_query.strip()}, as_node="grade_docs")
                    st.session_state.ui_chat_history.append(("assistant", f"✍️ Supervisor updated query to: '{override_query.strip()}'"))
                
                # Resume processing from the checkpoint memory
                with st.spinner("Executing live web search & compiling response..."):
                    for event in agent_graph.stream(None, st.session_state.graph_config, stream_mode="values"):
                        pass
                
                final_state = agent_graph.get_state(st.session_state.graph_config)
                ans = final_state.values.get("generation", "Error compiling response.")
                st.session_state.ui_chat_history.append(("assistant", ans))
                st.session_state.awaiting_approval = False
                st.rerun()
                
            if deny:
                agent_graph.update_state(st.session_state.graph_config, {"generation": "Web search denied by human supervisor."}, as_node="grade_docs")
                st.session_state.ui_chat_history.append(("assistant", "❌ Web search denied by human supervisor."))
                st.session_state.awaiting_approval = False
                st.rerun()

    # Normal Chat Input Element (Disabled when waiting for human input to prevent overlapping runs)
    if not st.session_state.awaiting_approval:
        if user_input := st.chat_input("Ask a question..."):
            st.session_state.ui_chat_history.append(("user", user_input))
            
            # Start running the compiled state graph engine
            with st.spinner("Analyzing document database structures..."):
                for event in agent_graph.stream({"question": user_input}, st.session_state.graph_config, stream_mode="values"):
                    pass
            
            # Evaluate why the stream stopped
            snapshot = agent_graph.get_state(st.session_state.graph_config)
            if snapshot.next:
                # We hit the web_search breakpoint tripwire!
                st.session_state.awaiting_approval = True
                st.rerun()
            else:
                # Completed successfully within the bounds of ChromaDB data
                ans = snapshot.values.get("generation", "No answer compiled.")
                st.session_state.ui_chat_history.append(("assistant", ans))
                st.rerun()

# --- RIGHT COLUMN: 3D SPACE GRAPH METRICS ---
with visual_col:
    st.subheader("Data Cluster Visualization")
    try:
        raw_data = db._collection.get(include=["documents", "embeddings"])
        total_chunks = len(raw_data["ids"])
        
        # Pull coordinates out via standard PCA projection
        pca = PCA(n_components=3)
        compressed = pca.fit_transform(np.array(raw_data["embeddings"]))
        
        df = pd.DataFrame({
            "Snippet": [doc[:70] + "..." for doc in raw_data["documents"]],
            "X": compressed[:, 0], "Y": compressed[:, 1], "Z": compressed[:, 2]
        })
        
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", hover_data=["Snippet"], template="plotly_dark")
        fig.update_traces(marker=dict(size=6, color="#636EFA", opacity=0.8))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode="cube"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization rendering error: {e}")
