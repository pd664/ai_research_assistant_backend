from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from app.pipeline.upload import upload_doc
from app.orchestrator.agent_controller import agent_controller
from pydantic import BaseModel
from app.observability.logger import log_run
import time 
from app.vector_store.vector_store import VectorStore
from app.utils.get_session_paths import get_session_paths

router = APIRouter()

class AskRequest(BaseModel):
    question: str

@router.post("/upload")
async def upload_file(request: Request,file: UploadFile=File(...)):
    session_id = request.headers.get("x-session-id")

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session ID")
    index_path, metadata_path = get_session_paths(session_id)
    print("index_path is ", index_path, metadata_path)
    
    vector_store = VectorStore(
        dim=1024,
        index_path=index_path,
        metadata_path=metadata_path
    )
    print(2)
    embedder = request.app.state.embedder
    print(3)
    try:
        start = time.time()
        print(4)
        response = await upload_doc(file, embedder, vector_store, session_id)
        print("Uploading File", "", start)
        return response
    except Exception as e:  
        print(f"Error occurred while uploading file: {e}")
        return {"error": str(e)}    

@router.post("/ask")
async def ask_question(request: Request, body: AskRequest):
    session_id = request.headers.get("x-session-id")

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session ID")
    index_path, metadata_path = get_session_paths(session_id)
    print("index_path is ", index_path, metadata_path)
    vector_store = VectorStore(
        dim=1024,
        index_path=index_path,
        metadata_path=metadata_path
    )
    
    embedder = request.app.state.embedder

    try:
        start = time.time()
        response = await agent_controller(body.question, embedder, vector_store)
        log_run("Answer", "", start)
        return response
    except Exception as e:
        print(f"Error occurred while processing question: {e}")
        return {"error": str(e)}
