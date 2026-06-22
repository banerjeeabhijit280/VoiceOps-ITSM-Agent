from itertools import chain
import os
import time
from typing import List, TypedDict
from pydantic import BaseModel, Field
import sounddevice as sd
import speech_recognition as sr
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import END, StateGraph

# PHASE 1: INGESTION (Run this to rebuild the DB with local embeddings)
############################################################################################################################################

def ingest_documents(pdf_path: str, csv_path: str):
    documents_to_ingest = []
    
    # --- PDF section ---
    # 1. Load the PDF
    pdf_loader = PyPDFLoader(pdf_path)
    docs = pdf_loader.load()
    print(f"Loaded PDF with {sum(len(doc.page_content) for doc in docs)} total characters.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Semantic Chunking
    chunker = SemanticChunker(
        embeddings
    )

    all_splits = chunker.split_documents(docs)
    print(f"Split document into {len(all_splits)} sub-documents.")

    # Add a metadata tag for the pdf chunks
    for doc in all_splits:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["sop_document"] = os.path.basename(pdf_path)

        if "page" in doc.metadata:
            doc.metadata["page_number"] = doc.metadata["page"]
        
    documents_to_ingest.extend(all_splits)
    # --- PDF section end ---
    
    # --- CSV section ---

    df = pd.read_csv(csv_path)

    csv_docs = []

    for _, row in df.iterrows():

        doc = Document(
            page_content=f"""
            Ticket ID: {row['TicketID']}
            Creation Date: {row['CreationDate']}
            Resolution Date: {row['ResolutionDate']}
            Status: {row['Status']}
            Category: {row['Category']}
            System: {row['System']}
            Version: {row['Version']}
            Issue Summary: {row['Issue_Summary']}
            Resolution Notes: {row['Resolution_Notes']}
            SOP Reference: {row['SOP_Reference']}
            """,

            metadata={
                "source_type": "csv",
                "ticket_id": str(row['TicketID']),
                "creation_date": str(row['CreationDate']),
                "resolution_date": str(row['ResolutionDate']),
                "status": str(row['Status']),
                "category": str(row['Category']),
                "system": str(row['System']),
                "version": str(row['Version']),
                "sop_reference": str(row['SOP_Reference'])
            }
        )

        csv_docs.append(doc)

    documents_to_ingest.extend(csv_docs)

    # --- CSV section end ---

    print(f"Total documents ready for ingestion: {len(documents_to_ingest)}") 

    # 3. Create the Chroma Store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="/tmp/VoiceOps_chroma_db" 
    )

    batch_size = 50  # Local processing allows much larger batches
    document_ids = []


    for i in range(0, len(documents_to_ingest), batch_size):
        batch = documents_to_ingest[i : i + batch_size]
        batch_ids = vector_store.add_documents(documents=batch)
        document_ids.extend(batch_ids)
        print(f"Processed batch {i//batch_size + 1}/{(len(documents_to_ingest) + batch_size - 1)//batch_size}")

    print(f"\nSuccessfully ingested {len(document_ids)} chunks into the local vector store!")

# UNCOMMENT THIS (below) THE FIRST TIME TO CONVERT THE PDF TO CREATE NEW LOCAL EMBEDDINGS:
#ingest_documents("ITSM Agent Data Generation - Google Gemini.pdf", "historical_itsm_tickets.csv")
############################################################################################################################################

# =========================
# GLOBAL SERVICES (Initialize once for speed)
print("Initializing LLM and Vector Store...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    persist_directory="/tmp/VoiceOps_chroma_db",
    embedding_function=embeddings
)

# Set up retriever and LLM once
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatOllama(model="mistral", temperature=0)
# =========================

class GraphState(TypedDict):
    question: str
    route: str
    documents: List[Document]
    generation: str
    
#"""These 2 agents are using the same DB, so they are using the filter in the metadata for differentiating between csv and pdf.
#The metedata was set during the data embedding process"""
# =========================
#Agent for pdf retrieval
def SOPRetriever(state: GraphState):
    print("Searching responses in the pdfs - SOP search")
    question = state["question"]
    # docs = retriever.invoke(question)
    sop_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"source_type": "pdf"}
        }
    )
    docs = sop_retriever.invoke(question)

    return {"documents": docs}
    # return {"documents": docs}
# =========================

# =========================
#Agent for csv retrieval
def ticketRetriever(state: GraphState):
    print("Searching responses in the csv - Ticket search.")
    question = state["question"]
    # docs = retriever.invoke(question)
    ticket_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"source_type": "csv"}
        }
    )
    docs = ticket_retriever.invoke(question)
    return {"documents": docs}
# =========================

# =========================
#Agent for both retrieval
def bothRetriever(state: GraphState):
    print("Searching both SOP and Ticket sources")

    pdf_docs = SOPRetriever(state)["documents"]
    ticket_docs = ticketRetriever(state)["documents"]

    return {
        "documents": pdf_docs + ticket_docs
    }
# =========================

# =========================
# CONFIG
SAMPLE_RATE = 16000
RECORD_DURATION = 5
# =========================

# =========================
# SPEECH TO TEXT (STT)
def listen():
    print("\n Speak now...") 
    try:
        recording = sd.rec( 
            int(RECORD_DURATION * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, dtype="int16" 
            ) 
        sd.wait() 
        audio_bytes = recording.tobytes() 
        audio_data = sr.AudioData( 
            audio_bytes, 
            sample_rate=SAMPLE_RATE, 
            sample_width=2 ) 
        recognizer = sr.Recognizer() 
        text = recognizer.recognize_google(audio_data) 
        print(f"\nYou: {text}") 
        return text 
    except Exception as e: 
        print(f"Speech recognition skipped/failed: {e}") 
        return None
# =========================

# =====================================
# TEXT TO SPEECH (TTS)
def speak(text):

    print(f"\n🤖 AI: {text}")

    safe_text = text.replace('"', '\\"')

    os.system(
        f'say -r 220 "{safe_text}"'
    )
# =====================================

# =========================
# Agent Supervisor
from pydantic import BaseModel, Field

class Route(BaseModel):
    route: str = Field(
        description="Route to use: 'sop' or 'ticket' or 'both'."
    )

# =========================
#supervisor node
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

            Return only one route:
            sop
            or
            ticket
            or
            both
            """
        ),
        ("human", "{user_question}")
    ])

    llm_with_structured_output = llm.with_structured_output(Route)

    chain = prompt_template | llm_with_structured_output

    response = chain.invoke({
        "user_question": question
    })

    route_val = "ticket"  # default fallback
    if response:
        if isinstance(response, dict):
            route_val = response.get("route", "ticket")
        elif hasattr(response, "route"):
            route_val = response.route

    return {"route": route_val}
# =========================

# =========================
#Reranker node, to filter out only the relevant responses and remove the illigical retiieved info.
class RelevanceScore(BaseModel):
    score: int = Field(
        ge=1,
        le=5,
        description="Relevance score from 1 to 5."
    )

def reranker(state: GraphState):
    query = state["question"]
    docs = state["documents"]
    print("Ranking extracted responses...")
    llmrank = llm.with_structured_output(RelevanceScore)
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a Relevance Ranking Agent.
            Rate how relevant the retrieved information is to the user's question.
            1 = Irrelevant
            2 = Slightly Relevant
            3 = Relevant
            4 = Highly Relevant
            5 = Perfect Match
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
                "data": f"""
                Metadata: {doc.metadata}

                Content:
                {doc.page_content}
                """
            })
            if response.score >= 3:
                filtered_docs.append(doc)
        except Exception as e:
            print(f"Reranking failed: {e}")
    print(f"Kept {len(filtered_docs)} of {len(docs)} documents")
    return {"documents": filtered_docs}
# =========================

# =========================
#Final LLM response
def answerGen(state: GraphState):
    query = state["question"]
    info = state["documents"]
    print("answerGen block runnning....")#Debug added
    
    formatted_data = "\n\n".join(
        f"""
        Metadata: {doc.metadata}

        Content:
        {doc.page_content}
        """
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
    response = chain.invoke({
        "user_question": query,
        "data": formatted_data
    })
    print("AI response: ", response)
    speak(response)
    return {"generation": response}

# =========================

workflow = StateGraph(GraphState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("ticketRetriever", ticketRetriever)
workflow.add_node("SOPRetriever", SOPRetriever)
workflow.add_node("bothRetriever", bothRetriever)
workflow.add_node("reranker", reranker)
workflow.add_node("answerGen", answerGen)

#Routing logic
def pdfOrCsv(state: GraphState):
    route = state["route"].lower().strip()
    if route == "sop":
        print("--- ROUTING: pdf -> SOP ---")
        return "sop"
    elif route == "ticket":
        print("--- ROUTING: csv -> ticket ---")
        return "ticket"
    elif route == "both":
        print("--- ROUTING: both ---")
        return "both"

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(     
    "supervisor",
    pdfOrCsv,
    {
        "sop": "SOPRetriever",
        "ticket": "ticketRetriever",
        "both": "bothRetriever"
    }
)
workflow.add_edge("ticketRetriever", "reranker")
workflow.add_edge("SOPRetriever", "reranker")
workflow.add_edge("bothRetriever", "reranker")
workflow.add_edge("reranker", "answerGen")

workflow.add_edge("answerGen", END)
rag_app = workflow.compile()


# EDITTED BY AGENT: Run the voice operations listening and compile/invoke the LangGraph workflow
if __name__ == "__main__":
    # Record user's query via voice
    user_query = listen()
    
    # Fallback to keyboard input if voice recording fails or returns None (e.g. in headless environments)
    if not user_query:
        print("Voice input failed or was empty. Please type your query:")
        user_query = input("You (Type): ").strip()
        
    if user_query:
        response = rag_app.invoke({
            "question": user_query,
            "route": "",
            "documents": [],
            "generation": ""
        })
    else:
        print("No valid query provided. Exiting.")