import os
import shutil
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="RAG Book Assistant API")

# Enable CORS for Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = "chroma_db"
METADATA_FILE = os.path.join(DB_DIR, "metadata.json")

class QueryRequest(BaseModel):
    query: str
    api_key: str = None

def get_api_key(api_key_override: str = None) -> str:
    key = api_key_override or os.getenv("MISTRAL_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="Mistral API Key is missing.")
    return key

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_metadata(metadata: dict):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

@app.get("/api/status")
def get_status():
    processed = os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0
    current_file = None
    if processed:
        metadata = load_metadata()
        current_file = metadata.get("current_file")
    api_key_configured = bool(os.getenv("MISTRAL_API_KEY", ""))
    return {
        "processed": processed,
        "current_file": current_file,
        "api_key_configured": api_key_configured
    }

@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    api_key: str = Form(None)
):
    key_to_use = get_api_key(api_key)
    
    # 1. Save uploaded file to temp file
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
        
    try:
        # 2. Clear old collection
        if os.path.exists(DB_DIR):
            try:
                temp_embeddings = MistralAIEmbeddings(
                    model="mistral-embed",
                    mistral_api_key=key_to_use
                )
                vectorstore = Chroma(
                    persist_directory=DB_DIR,
                    embedding_function=temp_embeddings
                )
                vectorstore.delete_collection()
            except Exception:
                pass
            
            # Clean up files inside chroma_db to prevent locking/bloating
            try:
                shutil.rmtree(DB_DIR)
            except Exception:
                pass
                
        os.makedirs(DB_DIR, exist_ok=True)
        
        # 3. Load PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        # 4. Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        
        # 5. Embedding Model
        embedding_model = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=key_to_use
        )
        
        # 6. Store in Chroma
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        
        # 7. Save metadata
        save_metadata({"current_file": file.filename})
        
        return {"success": True, "current_file": file.filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/api/query")
def query_assistant(request: QueryRequest):
    key_to_use = get_api_key(request.api_key)
    
    if not os.path.exists(DB_DIR) or len(os.listdir(DB_DIR)) == 0:
        raise HTTPException(status_code=400, detail="Database is empty. Please process a document first.")
        
    try:
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=key_to_use
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
            mistral_api_key=key_to_use
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant.\n\nUse only the provided context to answer the question. Please cite the page number(s) (e.g., [Page X]) in your answer whenever possible based on the context.\n\nIf the answer is not present in the context, say: \"I could not find the answer in the document.\""
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
        ])
        
        # 1. Retrieve matching chunks
        docs = retriever.invoke(request.query)
        
        # 2. Compile context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Build Prompt & Query Model
        final_prompt = prompt.invoke({
            "context": context,
            "question": request.query
        })
        response = llm.invoke(final_prompt)
        
        # 4. Format source documents
        retrieved_docs = [
            {"page_content": d.page_content, "metadata": d.metadata} 
            for d in docs
        ]
        
        return {
            "answer": response.content,
            "docs": retrieved_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying assistant: {str(e)}")

@app.post("/api/clear")
def clear_database(api_key: dict = None):
    # Retrieve API key override if passed
    key_override = None
    if api_key and "api_key" in api_key:
        key_override = api_key["api_key"]
    
    key_to_use = os.getenv("MISTRAL_API_KEY", "")
    if key_override:
        key_to_use = key_override
        
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
        except Exception:
            pass
            
    # Remove directory to clean database files
    try:
        shutil.rmtree(DB_DIR)
    except Exception:
        pass
        
    return {"processed": False, "current_file": None}
