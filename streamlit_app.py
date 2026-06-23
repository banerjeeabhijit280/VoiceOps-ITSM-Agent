"""
VoiceOps ITSM Intelligence Agent — Streamlit Demo UI
=====================================================
A premium, interactive demo interface for the multi-agent RAG pipeline.
"""

import streamlit as st
import pandas as pd
import time
import os
import io
import speech_recognition as sr
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceOps ITSM Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS — Premium dark glassmorphism theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default header/footer ── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0B0F19; }
::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 8px; }

/* ── Hero section ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7C3AED, #06B6D4, #10B981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94A3B8;
    font-weight: 400;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.3);
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.08);
}

/* ── Pipeline step cards ── */
.step-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: all 0.3s ease;
}
.step-card.active {
    border-color: #7C3AED;
    background: rgba(124, 58, 237, 0.08);
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.1);
}
.step-card.done {
    border-color: #10B981;
    background: rgba(16, 185, 129, 0.06);
}
.step-icon {
    font-size: 1.5rem;
    min-width: 2rem;
    text-align: center;
}
.step-label {
    font-weight: 600;
    font-size: 0.95rem;
    color: #E2E8F0;
}
.step-detail {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 2px;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(6,182,212,0.08));
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 1rem;
    line-height: 1.7;
    font-size: 1rem;
    color: #E2E8F0;
}
.answer-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7C3AED;
    margin-bottom: 0.75rem;
}

/* ── Document cards ── */
.doc-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #CBD5E1;
    transition: border-color 0.3s;
}
.doc-card:hover {
    border-color: rgba(6, 182, 212, 0.4);
}
.doc-meta {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.meta-badge {
    background: rgba(124, 58, 237, 0.15);
    color: #A78BFA;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.meta-badge.green {
    background: rgba(16, 185, 129, 0.15);
    color: #6EE7B7;
}
.meta-badge.blue {
    background: rgba(6, 182, 212, 0.15);
    color: #67E8F9;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: #0D1220 !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748B;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}
.tech-pill {
    display: inline-block;
    background: rgba(124, 58, 237, 0.12);
    color: #A78BFA;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px 3px;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.7rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* ── Example query chips ── */
.example-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    color: #CBD5E1;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 0.4rem;
    display: block;
    width: 100%;
    text-align: left;
}

/* ── Divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.3), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

/* ── Chat messages ── */
.stChatMessage {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
}

/* ── Voice input section ── */
.voice-section {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(6,182,212,0.05));
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.voice-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.voice-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #E2E8F0;
}
.voice-subtitle {
    font-size: 0.75rem;
    color: #94A3B8;
}
.voice-transcription {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.9rem;
    color: #6EE7B7;
}
.voice-error {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: #FCA5A5;
}
.input-mode-tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []
if "voice_query" not in st.session_state:
    st.session_state.voice_query = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None


# ─────────────────────────────────────────────────────────
# INITIALIZE SERVICES (cached so they load once)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_services():
    """Initialize Ollama models and ChromaDB vector store."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory="/tmp/VoiceOps_chroma_db",
        embedding_function=embeddings,
    )
    llm = ChatOllama(model="mistral", temperature=0)
    return embeddings, vector_store, llm


@st.cache_data(show_spinner=False)
def load_ticket_data():
    """Load the CSV for sidebar stats."""
    csv_path = os.path.join(os.path.dirname(__file__), "historical_itsm_tickets.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────
class Route(BaseModel):
    route: str = Field(description="Route to use: 'sop' or 'ticket' or 'both'.")

class RelevanceScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Relevance score from 1 to 5.")


# ─────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    route: str
    documents: List[Document]
    generation: str


# ─────────────────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────────────────
def build_agents(vector_store, llm):
    """Build all agent node functions using the shared vector_store & llm."""

    def supervisor(state: GraphState):
        question = state["question"]
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a routing agent.
                Determine whether the user's question should be answered from:
                - sop : SOP manuals and procedures
                - ticket : historical ticket records and incident history
                - both : if the question requires information from both sources
                Return only one route: sop or ticket or both
                """
            ),
            ("human", "{user_question}")
        ])
        llm_with_structured_output = llm.with_structured_output(Route)
        chain = prompt_template | llm_with_structured_output
        response = chain.invoke({"user_question": question})

        route_val = "ticket"  # default fallback
        if response:
            if isinstance(response, dict):
                route_val = response.get("route", "ticket")
            elif hasattr(response, "route"):
                route_val = response.route
        return {"route": route_val}

    def SOPRetriever(state: GraphState):
        question = state["question"]
        sop_retriever = vector_store.as_retriever(
            search_kwargs={"k": 4, "filter": {"source_type": "pdf"}}
        )
        docs = sop_retriever.invoke(question)
        return {"documents": docs}

    def ticketRetriever(state: GraphState):
        question = state["question"]
        ticket_retriever = vector_store.as_retriever(
            search_kwargs={"k": 4, "filter": {"source_type": "csv"}}
        )
        docs = ticket_retriever.invoke(question)
        return {"documents": docs}

    def bothRetriever(state: GraphState):
        pdf_docs = SOPRetriever(state)["documents"]
        ticket_docs = ticketRetriever(state)["documents"]
        return {"documents": pdf_docs + ticket_docs}

    def reranker(state: GraphState):
        query = state["question"]
        docs = state["documents"]
        llmrank = llm.with_structured_output(RelevanceScore)
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a Relevance Ranking Agent.
                Rate how relevant the retrieved information is to the user's question.
                1 = Irrelevant, 2 = Slightly Relevant, 3 = Relevant,
                4 = Highly Relevant, 5 = Perfect Match
                Retrieved Information:
                {data}
                """
            ),
            ("human", "{user_question}")
        ])
        chain = prompt_template | llmrank
        filtered_docs = []
        for doc in docs:
            try:
                response = chain.invoke({
                    "user_question": query,
                    "data": f"Metadata: {doc.metadata}\n\nContent:\n{doc.page_content}"
                })
                if response.score >= 3:
                    filtered_docs.append(doc)
            except Exception:
                pass
        return {"documents": filtered_docs}

    def answerGen(state: GraphState):
        query = state["question"]
        info = state["documents"]
        formatted_data = "\n\n".join(
            f"Metadata: {doc.metadata}\n\nContent:\n{doc.page_content}"
            for doc in info
        )
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an IT Support Agent.
                Answer ONLY using the retrieved information.
                Some retrieved documents may be irrelevant.
                Focus only on information relevant to the user's question.
                If the retrieved information does not contain the answer,
                say that the information was not found.

                Retrieved Information:
                {data}
                """
            ),
            ("human", "{user_question}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"user_question": query, "data": formatted_data})
        return {"generation": response}

    return supervisor, SOPRetriever, ticketRetriever, bothRetriever, reranker, answerGen


def build_graph(supervisor, SOPRetriever, ticketRetriever, bothRetriever, reranker, answerGen):
    """Compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("ticketRetriever", ticketRetriever)
    workflow.add_node("SOPRetriever", SOPRetriever)
    workflow.add_node("bothRetriever", bothRetriever)
    workflow.add_node("reranker", reranker)
    workflow.add_node("answerGen", answerGen)

    def pdfOrCsv(state: GraphState):
        route = state["route"].lower().strip()
        if route == "sop":
            return "sop"
        elif route == "ticket":
            return "ticket"
        else:
            return "both"

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor", pdfOrCsv,
        {"sop": "SOPRetriever", "ticket": "ticketRetriever", "both": "bothRetriever"}
    )
    workflow.add_edge("ticketRetriever", "reranker")
    workflow.add_edge("SOPRetriever", "reranker")
    workflow.add_edge("bothRetriever", "reranker")
    workflow.add_edge("reranker", "answerGen")
    workflow.add_edge("answerGen", END)

    return workflow.compile()


# ─────────────────────────────────────────────────────────
# RUN PIPELINE WITH UI UPDATES
# ─────────────────────────────────────────────────────────
def run_pipeline_with_ui(question: str, vector_store, llm):
    """Execute the pipeline step-by-step, updating Streamlit UI at each stage."""

    pipeline_log = {
        "question": question,
        "route": None,
        "docs_retrieved": 0,
        "docs_after_rerank": 0,
        "answer": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "timings": {},
    }

    # ── Step 1: Supervisor ──
    with st.status("🧠 **Supervisor** — Analyzing query intent...", expanded=True) as status:
        t0 = time.time()

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a routing agent.
            Determine whether the user's question should be answered from:
            - sop : SOP manuals and procedures
            - ticket : historical ticket records and incident history
            - both : if the question requires information from both sources
            Return only one route: sop or ticket or both"""),
            ("human", "{user_question}")
        ])
        llm_with_structured_output = llm.with_structured_output(Route)
        chain = prompt_template | llm_with_structured_output
        response = chain.invoke({"user_question": question})

        route_val = "ticket"
        if response:
            route_val = response.route if hasattr(response, "route") else response.get("route", "ticket")

        t1 = time.time()
        pipeline_log["route"] = route_val
        pipeline_log["timings"]["supervisor"] = round(t1 - t0, 2)

        route_labels = {
            "sop": "📄 SOP Documents",
            "ticket": "🎫 Historical Tickets",
            "both": "📄🎫 Both Sources"
        }
        st.markdown(f"""
        <div class="step-card done">
            <span class="step-icon">🧠</span>
            <div>
                <div class="step-label">Route Decision: {route_labels.get(route_val, route_val)}</div>
                <div class="step-detail">Completed in {pipeline_log['timings']['supervisor']}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        status.update(label=f"🧠 **Supervisor** — Routed to **{route_val.upper()}**", state="complete")

    # ── Step 2: Retrieval ──
    retriever_labels = {"sop": "📄 SOP Retriever", "ticket": "🎫 Ticket Retriever", "both": "📄🎫 Both Retrievers"}
    with st.status(f"{retriever_labels.get(route_val, '📦 Retriever')} — Searching knowledge base...", expanded=True) as status:
        t0 = time.time()

        docs = []
        if route_val == "sop" or route_val == "both":
            sop_retriever = vector_store.as_retriever(
                search_kwargs={"k": 4, "filter": {"source_type": "pdf"}}
            )
            docs.extend(sop_retriever.invoke(question))

        if route_val == "ticket" or route_val == "both":
            ticket_retriever = vector_store.as_retriever(
                search_kwargs={"k": 4, "filter": {"source_type": "csv"}}
            )
            docs.extend(ticket_retriever.invoke(question))

        t1 = time.time()
        pipeline_log["docs_retrieved"] = len(docs)
        pipeline_log["retrieved_docs"] = docs
        pipeline_log["timings"]["retrieval"] = round(t1 - t0, 2)

        # Show retrieved doc previews
        source_counts = {}
        for d in docs:
            src = d.metadata.get("source_type", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        source_summary = ", ".join(f"{v} {k.upper()}" for k, v in source_counts.items())

        st.markdown(f"""
        <div class="step-card done">
            <span class="step-icon">📦</span>
            <div>
                <div class="step-label">Retrieved {len(docs)} documents ({source_summary})</div>
                <div class="step-detail">Completed in {pipeline_log['timings']['retrieval']}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        status.update(label=f"📦 **Retrieval** — Found **{len(docs)} documents**", state="complete")

    # ── Step 3: Reranking ──
    with st.status("🏆 **Reranker** — Scoring document relevance...", expanded=True) as status:
        t0 = time.time()

        llmrank = llm.with_structured_output(RelevanceScore)
        rank_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Relevance Ranking Agent.
            Rate how relevant the retrieved information is to the user's question.
            1 = Irrelevant, 2 = Slightly Relevant, 3 = Relevant,
            4 = Highly Relevant, 5 = Perfect Match
            Retrieved Information:
            {data}"""),
            ("human", "{user_question}")
        ])
        rank_chain = rank_prompt | llmrank

        filtered_docs = []
        progress_bar = st.progress(0, text="Scoring documents...")
        for i, doc in enumerate(docs):
            try:
                resp = rank_chain.invoke({
                    "user_question": question,
                    "data": f"Metadata: {doc.metadata}\n\nContent:\n{doc.page_content}"
                })
                if resp.score >= 3:
                    filtered_docs.append(doc)
            except Exception:
                pass
            progress_bar.progress((i + 1) / max(len(docs), 1), text=f"Scored {i+1}/{len(docs)} documents")

        t1 = time.time()
        pipeline_log["docs_after_rerank"] = len(filtered_docs)
        pipeline_log["reranked_docs"] = filtered_docs
        pipeline_log["timings"]["reranking"] = round(t1 - t0, 2)

        st.markdown(f"""
        <div class="step-card done">
            <span class="step-icon">🏆</span>
            <div>
                <div class="step-label">Kept {len(filtered_docs)} of {len(docs)} documents (score ≥ 3)</div>
                <div class="step-detail">Completed in {pipeline_log['timings']['reranking']}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        status.update(label=f"🏆 **Reranker** — Kept **{len(filtered_docs)}/{len(docs)}** relevant docs", state="complete")

    # ── Step 4: Answer Generation ──
    with st.status("💬 **Answer Generator** — Synthesizing response...", expanded=True) as status:
        t0 = time.time()

        formatted_data = "\n\n".join(
            f"Metadata: {doc.metadata}\n\nContent:\n{doc.page_content}"
            for doc in filtered_docs
        )
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an IT Support Agent.
            Answer ONLY using the retrieved information.
            Some retrieved documents may be irrelevant.
            Focus only on information relevant to the user's question.
            If the retrieved information does not contain the answer,
            say that the information was not found.

            Retrieved Information:
            {data}"""),
            ("human", "{user_question}")
        ])
        gen_chain = gen_prompt | llm | StrOutputParser()
        answer = gen_chain.invoke({"user_question": question, "data": formatted_data})

        t1 = time.time()
        pipeline_log["answer"] = answer
        pipeline_log["timings"]["generation"] = round(t1 - t0, 2)

        st.markdown(f"""
        <div class="step-card done">
            <span class="step-icon">💬</span>
            <div>
                <div class="step-label">Answer generated successfully</div>
                <div class="step-detail">Completed in {pipeline_log['timings']['generation']}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        status.update(label="💬 **Answer Generator** — Response ready!", state="complete")

    return pipeline_log


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo & Title
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <div style="font-size:2.5rem; margin-bottom:0.25rem;">🎙️</div>
            <div style="font-size:1.2rem; font-weight:700; 
                        background: linear-gradient(135deg, #7C3AED, #06B6D4);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text;">
                VoiceOps ITSM Agent
            </div>
            <div style="font-size:0.75rem; color:#64748B; margin-top:0.25rem;">
                Multi-Agent RAG Pipeline Demo
            </div>
        </div>
        <hr style="border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(124,58,237,0.3),transparent); margin:1rem 0;">
        """, unsafe_allow_html=True)

        # Architecture
        st.markdown('<div class="sidebar-section-title">🏗️ Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        ```
        Voice/Text ──▶ Supervisor
                         │
                ┌────────┼────────┐
                ▼        ▼        ▼
              SOP     Ticket    Both
            Retriever Retriever Retriever
                └────────┼────────┘
                         ▼
                      Reranker
                         ▼
                    Answer Gen
                         ▼
                    🔊 Response
        ```
        """)

        # Tech Stack
        st.markdown('<div class="sidebar-section-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-wrap:wrap; gap:4px;">
            <span class="tech-pill">LangGraph</span>
            <span class="tech-pill">LangChain</span>
            <span class="tech-pill">Ollama</span>
            <span class="tech-pill">Mistral</span>
            <span class="tech-pill">ChromaDB</span>
            <span class="tech-pill">Nomic Embed</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Pydantic</span>
        </div>
        """, unsafe_allow_html=True)

        # Data Stats
        df = load_ticket_data()
        if not df.empty:
            st.markdown('<div class="sidebar-section-title">📊 Knowledge Base</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tickets", len(df))
            with col2:
                st.metric("Categories", df["Category"].nunique())

            col3, col4 = st.columns(2)
            with col3:
                st.metric("Systems", df["System"].nunique())
            with col4:
                st.metric("SOPs", df["SOP_Reference"].nunique())

        # Example Queries
        st.markdown('<div class="sidebar-section-title">💡 Example Queries</div>', unsafe_allow_html=True)

        examples = [
            "How do I fix VPN Error 45?",
            "Show me Active Directory lockout tickets",
            "What is the SOP for PostgreSQL migration?",
            "How to resolve WebLogic ADMIN state issue?",
            "What is the firmware upgrade path for Cisco Nexus?",
        ]

        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state["prefill_query"] = example
                st.rerun()

        # Footer
        st.markdown("""
        <hr style="border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(124,58,237,0.3),transparent); margin:1.5rem 0 1rem 0;">
        <div style="text-align:center; font-size:0.7rem; color:#475569;">
            Built with LangChain · LangGraph · Ollama<br/>
            100% Local · Zero Cloud Dependencies
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Hero
    st.markdown("""
    <div class="hero-container animate-in">
        <div class="hero-title">VoiceOps ITSM Intelligence Agent</div>
        <div class="hero-subtitle">
            Ask IT support questions using natural language. The multi-agent pipeline
            routes your query, retrieves from SOP docs & historical tickets, re-ranks 
            for relevance, and generates a grounded answer — all running locally.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics bar
    total_time = sum(
        log.get("timings", {}).values()
        for log in st.session_state.pipeline_logs
        for _ in [None]
    ) if st.session_state.pipeline_logs else 0

    if st.session_state.pipeline_logs:
        last_log = st.session_state.pipeline_logs[-1]
        last_total = sum(last_log.get("timings", {}).values())
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.messages) // 2}</div>
                <div class="metric-label">Queries Processed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_log.get('route', '—').upper()}</div>
                <div class="metric-label">Last Route</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{last_log.get('docs_after_rerank', 0)}/{last_log.get('docs_retrieved', 0)}</div>
                <div class="metric-label">Docs Kept/Retrieved</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{round(last_total, 1)}s</div>
                <div class="metric-label">Pipeline Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # ── Voice / Text input mode ──
    input_mode = st.radio(
        "Input Mode",
        ["⌨️ Text", "🎤 Voice"],
        horizontal=True,
        label_visibility="collapsed",
    )

    query = None

    if input_mode == "🎤 Voice":
        st.markdown("""
        <div class="voice-section">
            <div class="voice-header">
                <span style="font-size:1.3rem;">🎤</span>
                <div>
                    <div class="voice-title">Voice Input</div>
                    <div class="voice-subtitle">Click the microphone to record your question, then stop to transcribe</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        audio_data = st.audio_input("Record your IT support question", key="voice_recorder")

        if audio_data is not None:
            # Generate a unique ID for this audio to avoid re-processing
            audio_id = id(audio_data)
            if audio_id != st.session_state.last_audio_id:
                st.session_state.last_audio_id = audio_id
                with st.spinner("🔄 Transcribing audio..."):
                    try:
                        recognizer = sr.Recognizer()
                        audio_bytes = audio_data.read()
                        audio_file = io.BytesIO(audio_bytes)
                        with sr.AudioFile(audio_file) as source:
                            audio = recognizer.record(source)
                        transcribed_text = recognizer.recognize_google(audio)
                        st.session_state.voice_query = transcribed_text
                    except sr.UnknownValueError:
                        st.markdown("""
                        <div class="voice-error">
                            ⚠️ Could not understand the audio. Please try speaking more clearly.
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.voice_query = None
                    except sr.RequestError as e:
                        st.markdown(f"""
                        <div class="voice-error">
                            ⚠️ Speech recognition service error: {e}
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.voice_query = None
                    except Exception as e:
                        st.markdown(f"""
                        <div class="voice-error">
                            ⚠️ Audio processing error: {e}<br/>
                            Tip: Make sure the recording is clear and not too short.
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.voice_query = None

            if st.session_state.voice_query:
                st.markdown(f"""
                <div class="voice-transcription">
                    ✅ <strong>Transcribed:</strong> "{st.session_state.voice_query}"
                </div>
                """, unsafe_allow_html=True)

                if st.button("🚀 Submit Voice Query", type="primary", use_container_width=True):
                    query = st.session_state.voice_query
                    st.session_state.voice_query = None
                    st.session_state.last_audio_id = None

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Handle prefilled query from sidebar example buttons
    prefill = st.session_state.pop("prefill_query", None)

    # Text input (only in text mode)
    if input_mode == "⌨️ Text":
        text_query = st.chat_input("Ask an IT support question...", key="chat_input")
        if text_query:
            query = text_query
    if prefill:
        query = prefill

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Initialize services
        try:
            embeddings, vector_store, llm = init_services()
        except Exception as e:
            st.error(f"""
            **⚠️ Service Initialization Failed**
            
            Make sure Ollama is running and the required models are pulled:
            ```bash
            ollama serve
            ollama pull mistral
            ollama pull nomic-embed-text
            ```
            
            Error: `{e}`
            """)
            return

        # Run pipeline
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("##### ⚙️ Pipeline Execution")
            try:
                pipeline_log = run_pipeline_with_ui(query, vector_store, llm)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

            # Display answer
            st.markdown(f"""
            <div class="answer-box animate-in">
                <div class="answer-label">🤖 AI Response</div>
                {pipeline_log['answer']}
            </div>
            """, unsafe_allow_html=True)

            # Expandable: Retrieved Documents
            if pipeline_log.get("reranked_docs"):
                with st.expander(f"📚 View Retrieved Documents ({len(pipeline_log['reranked_docs'])} relevant)", expanded=False):
                    for i, doc in enumerate(pipeline_log["reranked_docs"]):
                        source_type = doc.metadata.get("source_type", "unknown")
                        badge_class = "green" if source_type == "pdf" else "blue"
                        badge_label = "SOP" if source_type == "pdf" else "TICKET"

                        extra_badges = ""
                        if "ticket_id" in doc.metadata:
                            extra_badges += f'<span class="meta-badge">{doc.metadata["ticket_id"]}</span>'
                        if "category" in doc.metadata:
                            extra_badges += f'<span class="meta-badge green">{doc.metadata["category"]}</span>'
                        if "system" in doc.metadata:
                            extra_badges += f'<span class="meta-badge blue">{doc.metadata["system"]}</span>'
                        if "sop_document" in doc.metadata:
                            extra_badges += f'<span class="meta-badge">{doc.metadata["sop_document"][:30]}</span>'

                        content_preview = doc.page_content[:300].replace("\n", " ").strip()

                        st.markdown(f"""
                        <div class="doc-card">
                            <div class="doc-meta">
                                <span class="meta-badge {badge_class}">{badge_label}</span>
                                {extra_badges}
                            </div>
                            <div>{content_preview}{'...' if len(doc.page_content) > 300 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Expandable: Timing breakdown
            with st.expander("⏱️ Performance Breakdown", expanded=False):
                timings = pipeline_log.get("timings", {})
                timing_data = {
                    "Stage": ["🧠 Supervisor", "📦 Retrieval", "🏆 Reranking", "💬 Generation"],
                    "Time (s)": [
                        timings.get("supervisor", 0),
                        timings.get("retrieval", 0),
                        timings.get("reranking", 0),
                        timings.get("generation", 0),
                    ]
                }
                st.dataframe(
                    pd.DataFrame(timing_data),
                    hide_index=True,
                    use_container_width=True
                )
                st.markdown(f"**Total pipeline time:** {round(sum(timings.values()), 2)}s")

            # Save to session state
            st.session_state.messages.append({"role": "assistant", "content": pipeline_log["answer"]})
            st.session_state.pipeline_logs.append(pipeline_log)


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
