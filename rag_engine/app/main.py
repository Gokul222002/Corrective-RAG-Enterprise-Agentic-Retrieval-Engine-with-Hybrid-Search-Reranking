import time
from pathlib import Path
from typing import Optional, List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag_engine.app.config import settings, BASE_DIR
from rag_engine.app.schemas.api_models import (
    IngestTextRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceChunk
)
from rag_engine.app.services.ingestion import ingestion_pipeline
from rag_engine.app.services.retriever import hybrid_retriever
from rag_engine.app.core.graph import crag_graph
from rag_engine.app.core.state import GraphState

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup event: Seeds sample knowledge base if empty."""
    sample_dir = BASE_DIR / "rag_engine" / "sample_docs"
    if not sample_dir.exists():
        sample_dir = Path("sample_docs")

    # If database is fresh or empty, auto-seed sample documents
    if hybrid_retriever.count() == 0 and sample_dir.exists():
        print("[Startup] Empty corpus detected. Automatically seeding enterprise sample documents...")
        for doc_file in sample_dir.glob("*.md"):
            try:
                ingestion_pipeline.ingest_file(doc_file)
                print(f"[Startup] Ingested sample: {doc_file.name}")
            except Exception as e:
                print(f"[Startup Warning] Could not ingest {doc_file.name}: {e}")
    else:
        print(f"[Startup] Corpus loaded with {hybrid_retriever.count()} indexed chunks.")

    yield

app = FastAPI(
    title="Enterprise Corrective RAG (CRAG) Engine",
    description="Agentic Retrieval Engine with Hybrid Search (Qdrant + BM25) and FlashRank Reranking",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

TEMPLATES_DIR = Path(__file__).parent / "templates"

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/ui", response_class=HTMLResponse, tags=["Web UI"])
async def serve_interactive_web():
    """Serves the interactive CRAG Agentic Web Dashboard."""
    template_file = TEMPLATES_DIR / "index.html"
    if template_file.exists():
        return HTMLResponse(content=template_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Enterprise CRAG Engine API</h1><p>Visit <a href='/docs'>/docs</a> for OpenAPI documentation.</p>")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns real-time system status and index telemetry."""
    total_docs = hybrid_retriever.count()
    bm25_ready = hybrid_retriever.bm25_index is not None
    gemini_ready = bool(settings.GOOGLE_API_KEY)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        qdrant_status="connected (local)",
        total_documents_indexed=total_docs,
        bm25_initialized=bm25_ready,
        gemini_configured=gemini_ready
    )

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    request: Optional[IngestTextRequest] = None,
    file: Optional[UploadFile] = File(default=None)
):
    """
    Ingest documents into both Qdrant Vector Store and BM25 index.
    Accepts JSON body or multipart file upload (.txt, .md, .pdf).
    """
    try:
        if file is not None:
            content_bytes = await file.read()
            text_content = ingestion_pipeline.extract_text_from_bytes(content_bytes, file.filename)
            chunks = ingestion_pipeline.process_document(
                content=text_content,
                source=file.filename
            )
            return IngestResponse(
                status="success",
                chunks_indexed=len(chunks),
                source=file.filename,
                collection=settings.QDRANT_COLLECTION,
                chunk_ids=[c.metadata.get("chunk_id") for c in chunks]
            )

        if request is not None and request.text:
            chunks = ingestion_pipeline.process_document(
                content=request.text,
                source=request.source,
                extra_metadata=request.metadata
            )
            return IngestResponse(
                status="success",
                chunks_indexed=len(chunks),
                source=request.source,
                collection=settings.QDRANT_COLLECTION,
                chunk_ids=[c.metadata.get("chunk_id") for c in chunks]
            )

        raise HTTPException(status_code=400, detail="Must provide either a JSON payload with 'text' or an uploaded 'file'.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Reasoning"])
async def query_engine(payload: QueryRequest):
    """
    Executes the full Corrective RAG (CRAG) Agentic cycle:
    1. Hybrid Search (Dense Qdrant + Sparse BM25)
    2. Neural FlashRank Reranker (top-3 chunks)
    3. Document Relevance Grading
    4. Conditional Routing (Reformulation loop or grounded generation)
    """
    start_time = time.perf_counter()

    initial_state: GraphState = {
        "question": payload.query,
        "original_question": payload.query,
        "documents": [],
        "generation": "",
        "relevance_score": "no",
        "rewrite_count": 0,
        "rewritten_query": None,
        "route_history": [],
        "warning": None
    }

    try:
        final_state = crag_graph.invoke(initial_state)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Map document chunks to SourceChunk schema
        source_chunks: List[SourceChunk] = []
        for doc in final_state.get("documents", []):
            meta = doc.metadata or {}
            source_chunks.append(SourceChunk(
                content=doc.page_content,
                source=meta.get("source", "Unknown"),
                chunk_id=meta.get("chunk_id", "Unknown"),
                score=meta.get("rerank_score"),
                dense_score=meta.get("dense_score"),
                sparse_score=meta.get("sparse_score")
            ))

        return QueryResponse(
            answer=final_state.get("generation", "No answer generated."),
            sources=source_chunks,
            rewritten_query=final_state.get("rewritten_query"),
            relevance_score=final_state.get("relevance_score", "no"),
            rewrite_count=final_state.get("rewrite_count", 0),
            latency_ms=round(latency_ms, 2),
            route_history=final_state.get("route_history", []),
            warning=final_state.get("warning")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CRAG execution error: {str(e)}")

@app.post("/api/ingest-sample", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_sample_document(body: dict):
    """Convenience endpoint to index sample docs in sample_docs/."""
    sample_name = body.get("sample_file", "enterprise_crag_architecture.md")
    sample_path = BASE_DIR / "rag_engine" / "sample_docs" / sample_name
    
    if not sample_path.exists():
        sample_path = Path("sample_docs") / sample_name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample document '{sample_name}' not found.")

    chunks = ingestion_pipeline.ingest_file(sample_path)
    return IngestResponse(
        status="success",
        chunks_indexed=len(chunks),
        source=sample_name,
        collection=settings.QDRANT_COLLECTION,
        chunk_ids=[c.metadata.get("chunk_id") for c in chunks]
    )
