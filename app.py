import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Constants
DB_DIR = "chroma_db"

# --- Premium Minimalistic Theme & CSS Styling ---
st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium look
st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    /* Header Typography */
    .hero-title {
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.05em;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: -0.01em;
    }
    
    /* Premium Sidebar Cards */
    .sidebar-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .sidebar-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F3F4F6;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Clean Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.25);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(59, 130, 246, 0.35);
        color: #FFFFFF;
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* Danger Button Style */
    div[data-testid="stSidebar"] .stButton>button.danger-btn {
        background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.25);
    }
    
    div[data-testid="stSidebar"] .stButton>button.danger-btn:hover {
        box-shadow: 0 6px 14px rgba(239, 68, 68, 0.35);
    }
    
    /* Custom Chat Styling */
    .user-bubble {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px 12px 0 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        max-width: 80%;
        margin-left: auto;
    }
    
    .ai-bubble {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px 12px 12px 0;
        padding: 1rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    
    /* Citation Expander Box */
    .source-box {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #94A3B8;
    }
    
    .source-meta {
        font-weight: 600;
        color: #60A5FA;
        margin-bottom: 0.25rem;
    }
    
    /* Hide Default Streamlit Style Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- State Management Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Retrieve API key early for function access
env_api_key = os.getenv("MISTRAL_API_KEY", "")

# --- Helper Functions ---
def clear_vectorstore():
    """Removes the persistent Chroma collection and resets state."""
    # Obtain the API key from inputs/session if possible
    key_to_use = st.session_state.get("api_key", env_api_key)
    if os.path.exists(DB_DIR) and key_to_use:
        try:
            embedding_model = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=key_to_use
            )
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embedding_model
            )
            vectorstore.delete_collection()
        except Exception as e:
            st.sidebar.error(f"Error clearing collection: {e}")
    st.session_state.processed = False
    st.session_state.current_file = None
    st.session_state.chat_history = []
    st.rerun()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<div class="sidebar-card-title">⚙️ API Config</div>', unsafe_allow_html=True)
    
    api_key_input = st.text_input(
        "Mistral API Key",
        value=env_api_key,
        type="password",
        placeholder="Enter Mistral API Key...",
        help="If configured in .env file, this will populate automatically."
    )
    
    # Store in session state for reference in helper functions
    st.session_state["api_key"] = api_key_input if api_key_input else env_api_key
    mistral_api_key = st.session_state["api_key"]

    st.markdown('<hr style="border-color: #1F2937; margin: 1.5rem 0;" />', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card-title">📚 Document Ingestion</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF Book",
        type=["pdf"],
        help="Upload the book PDF you want to query."
    )
    
    # Ingestion Trigger Button
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        # If there's a new file, prompt user to ingest it
        is_new_file = st.session_state.current_file != file_name
        
        if is_new_file:
            st.info(f"New file '{file_name}' loaded. Click 'Process' to ingest.")
            
        process_button = st.button("⚡ Process Document")
        
        if process_button:
            if not mistral_api_key:
                st.error("Please provide a Mistral API Key to proceed.")
            else:
                try:
                    with st.spinner("Processing PDF... Please wait."):
                        # 1. Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # 2. Safely clear old collection data if database exists
                            if os.path.exists(DB_DIR):
                                try:
                                    temp_embeddings = MistralAIEmbeddings(
                                        model="mistral-embed",
                                        mistral_api_key=mistral_api_key
                                    )
                                    vectorstore = Chroma(
                                        persist_directory=DB_DIR,
                                        embedding_function=temp_embeddings
                                    )
                                    vectorstore.delete_collection()
                                except Exception:
                                    pass
                            
                            # 3. Load PDF
                            status_loader = st.empty()
                            status_loader.text("📖 Reading document pages...")
                            loader = PyPDFLoader(tmp_file_path)
                            docs = loader.load()
                            
                            # 4. Split PDF into chunks
                            status_loader.text("✂️ Splitting text into chunks...")
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200
                            )
                            chunks = splitter.split_documents(docs)
                            
                            # 5. Embedding Model
                            status_loader.text("🧠 Generating vector embeddings...")
                            embedding_model = MistralAIEmbeddings(
                                model="mistral-embed",
                                mistral_api_key=mistral_api_key
                            )
                            
                            # 6. Store in Chroma Database
                            status_loader.text("💾 Creating Chroma Vector Database...")
                            vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding=embedding_model,
                                persist_directory=DB_DIR
                            )
                            
                            status_loader.empty()
                            st.session_state.processed = True
                            st.session_state.current_file = file_name
                            st.session_state.chat_history = [] # Reset chat for new document
                            st.success("Database Created Successfully!")
                            st.rerun()
                            
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)
                                
                except Exception as e:
                    st.error(f"Failed to process document: {e}")
                    
    st.markdown('<hr style="border-color: #1F2937; margin: 1.5rem 0;" />', unsafe_allow_html=True)
    
    # Database status indicators & delete action
    if st.session_state.processed:
        st.markdown('<div class="sidebar-card-title" style="color: #10B981;">🟢 Active Book</div>', unsafe_allow_html=True)
        st.caption(f"Currently loaded: **{st.session_state.current_file or 'Pre-loaded DB'}**")
        
        # Apply special class to make button red
        st.markdown('<div class="danger-btn-wrapper">', unsafe_allow_html=True)
        if st.button("🗑️ Clear Database", key="clear_db_btn"):
            clear_vectorstore()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-card-title" style="color: #EF4444;">🔴 Database Status</div>', unsafe_allow_html=True)
        st.caption("No book loaded. Database is empty.")


# --- Main Application Area ---
st.markdown('<div class="hero-title">RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Ask questions and retrieve semantic context from your uploaded books using Mistral AI and Chroma DB.</div>', unsafe_allow_html=True)

# Main UI Logic
if not st.session_state.processed:
    # Empty State Display
    st.info("👋 Welcome! Please configure your Mistral API Key and upload a PDF book in the sidebar. Once you click 'Process Document', the assistant will parse, chunk, and embed the pages, enabling you to ask questions.")
else:
    # Set up LLM & Retriever elements
    try:
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=mistral_api_key
        )
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )
        llm = ChatMistralAI(
            model="mistral-small-2506",
            mistral_api_key=mistral_api_key
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant.\n\nUse only the provided context to answer the question.\n\nIf the answer is not present in the context, say: \"I could not find the answer in the document.\""
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
        ])
    except Exception as init_err:
        st.error(f"Error initializing AI components: {init_err}. Check your API Key settings.")
        st.stop()

    # Display Chat Messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                # Show citations in collapsible expander
                if msg.get("docs"):
                    with st.expander("📚 Retrieved Context & Sources", expanded=False):
                        for i, doc in enumerate(msg["docs"], 1):
                            doc_metadata = doc.get("metadata", {})
                            page_num = doc_metadata.get('page', 0) + 1  # 0-indexed page to 1-indexed display
                            page_content = doc.get("page_content", "")
                            st.markdown(f"""
                                <div class="source-box">
                                    <div class="source-meta">Chunk {i} - Page {page_num}</div>
                                    <div class="source-content">{page_content}</div>
                                </div>
                            """, unsafe_allow_html=True)

    # Handle New Chat Input
    query = st.chat_input("Ask a question about the document...")
    
    if query:
        # Display user message instantly
        st.chat_message("user").write(query)
        
        # Add to local state
        user_msg = {"role": "user", "content": query}
        st.session_state.chat_history.append(user_msg)
        
        # Process retrieval and answer generation
        with st.spinner("Searching document & generating answer..."):
            try:
                # 1. Retrieve matching document chunks
                docs = retriever.invoke(query)
                
                # 2. Compile context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # 3. Build Prompt & Query Model
                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })
                response = llm.invoke(final_prompt)
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.write(response.content)
                    with st.expander("📚 Retrieved Context & Sources", expanded=False):
                        for i, doc in enumerate(docs, 1):
                            page_num = doc.metadata.get('page', 0) + 1
                            st.markdown(f"""
                                <div class="source-box">
                                    <div class="source-meta">Chunk {i} - Page {page_num}</div>
                                    <div class="source-content">{doc.page_content}</div>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Add to chat history state
                ai_msg = {
                    "role": "assistant",
                    "content": response.content,
                    "docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
                }
                st.session_state.chat_history.append(ai_msg)
                
            except Exception as run_err:
                st.error(f"Error querying assistant: {run_err}")