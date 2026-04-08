from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from dotenv import load_dotenv
import os
import logging

from app.ingestion.embedder import Embedder
from app.vector_store.vector_store import VectorStore

# ------------------ Setup Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


load_dotenv()

INDEX_PATH = os.getenv("INDEX_PATH")
METADATA_PATH = os.getenv("METADATA_PATH")

def parse_allowed_origins():
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if not raw:
        logger.warning("ALLOWED_ORIGINS not set. Defaulting to empty list.")
        return []
    
    return [origin.strip() for origin in raw.split(",") if origin.strip()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Initializing application resources...")

        app.state.embedder = Embedder()

        logger.info("Application startup complete. Ready to serve requests.")

    except Exception as e:
        logger.exception("Failed during application startup.")
        raise RuntimeError("Startup failed. Check logs.") from e

    yield

    logger.info("Shutting down application...")

app = FastAPI(
    title="Your Project API",
    version="1.0.0",
    lifespan=lifespan
)

allowed_origins = parse_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(router)
