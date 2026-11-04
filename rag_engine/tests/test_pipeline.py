import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from rag_engine.app.config import settings
from rag_engine.app.main import app
from rag_engine.app.services.ingestion import ingestion_pipeline
from rag_engine.app.services.retriever import hybrid_retriever
from rag_engine.app.core.graph import crag_graph
from rag_engine.app.core.state import GraphState
from rag_engine.app.core.nodes import decide_next_step

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_corpus():
    """Seed test corpus with enterprise sample documents before running tests."""
    sample_dir = Path(__file__).resolve().parent.parent / "sample_docs"
    docs_to_seed = [
        "enterprise_crag_architecture.md",
        "distributed_consensus_raft.md",
        "kubernetes_pod_lifecycle.md"
    ]
    for doc_name in docs_to_seed:
        file_path = sample_dir / doc_name
        if file_path.exists():
            ingestion_pipeline.ingest_file(file_path)

def test_ingestion_and_chunking():
    """Verify recursive chunking, metadata enrichment, and index population."""
    sample_text = (
        "# Fault Tolerance\n\n"
        "Distributed consensus requires quorum agreement among nodes. "
        "Each node maintains an active state machine log and replicates updates.\n\n"
        "Heartbeats are transmitted at regular intervals to prevent split-brain conditions."
    )
    chunks = ingestion_pipeline.process_document(
        content=sample_text,
        source="test_consensus_spec.md"
    )

    assert len(chunks) >= 1
    first_chunk = chunks[0]
    assert first_chunk.metadata["source"] == "test_consensus_spec.md"
    assert "chunk_id" in first_chunk.metadata
    assert first_chunk.metadata["char_length"] <= settings.CHUNK_SIZE
    assert hybrid_retriever.count() > 0

def test_hybrid_retrieval_and_top_3_rerank():
    """Verify hybrid search retrieves candidates and reranker returns top-3 chunks."""
    query = "What is the role of heartbeats and quorum in distributed consensus?"
    top_chunks = hybrid_retriever.hybrid_retrieve_and_rerank(
        query=query,
        top_k=settings.TOP_K_RETRIEVAL,
        top_n=settings.TOP_N_RERANK
    )

    assert len(top_chunks) > 0
    assert len(top_chunks) <= settings.TOP_N_RERANK
    # Verify rerank scores are assigned
    for doc in top_chunks:
        assert "rerank_score" in doc.metadata

def test_exact_match_retrieval():
    """
    Verify exact match query:
    - Successfully retrieves relevant context
    - Relevance grader returns 'yes'
    - Routes directly to generate without entering query rewriter
    """
    query = "How does the Raft protocol elect a leader and achieve quorum commit?"
    initial_state: GraphState = {
        "question": query,
        "original_question": query,
        "documents": [],
        "generation": "",
        "relevance_score": "no",
        "rewrite_count": 0,
        "rewritten_query": None,
        "route_history": [],
        "warning": None
    }

    final_state = crag_graph.invoke(initial_state)

    assert final_state["relevance_score"] == "yes"
    assert final_state["rewrite_count"] == 0
    assert "generate" in final_state["route_history"]
    assert "transform_query" not in final_state["route_history"]
    assert len(final_state["documents"]) > 0
    assert "Raft" in final_state["generation"] or "leader" in final_state["generation"].lower()

def test_ambiguous_query_rewriter_trigger():
    """
    Verify ambiguous query behavior:
    - Irrelevant or low-signal query triggers the query-rewriter
    - Graph routes through 'transform_query' node
    - 'rewrite_count' is incremented
    - Search query is expanded with semantic keywords
    """
    ambiguous_query = "xyznonexistent terms and obscure failure"
    initial_state: GraphState = {
        "question": ambiguous_query,
        "original_question": ambiguous_query,
        "documents": [],
        "generation": "",
        "relevance_score": "no",
        "rewrite_count": 0,
        "rewritten_query": None,
        "route_history": [],
        "warning": None
    }

    final_state = crag_graph.invoke(initial_state)

    # Should have triggered transform_query at least once
    assert "transform_query" in final_state["route_history"]
    assert final_state["rewrite_count"] > 0
    assert final_state["rewritten_query"] is not None
    assert final_state["rewritten_query"] != ambiguous_query

def test_guardrail_max_rewrites_and_warning():
    """
    Verify guardrail limits:
    - Cycle does not loop infinitely
    - Maximum 2 rewrites are permitted
    - Gracefully exits to 'generate' with warning flag
    """
    nonsense_query = "zzqqxx1234987 nonsense non-indexed gibberish"
    initial_state: GraphState = {
        "question": nonsense_query,
        "original_question": nonsense_query,
        "documents": [],
        "generation": "",
        "relevance_score": "no",
        "rewrite_count": 0,
        "rewritten_query": None,
        "route_history": [],
        "warning": None
    }

    final_state = crag_graph.invoke(initial_state)

    # Max rewrites must not exceed settings.MAX_REWRITES (2)
    assert final_state["rewrite_count"] <= settings.MAX_REWRITES
    assert final_state["warning"] is not None
    assert "Insufficient context" in final_state["warning"]
    assert "generate" in final_state["route_history"]

def test_terminal_response_grounding_and_citation():
    """
    Verify grounded response generation:
    - Answers Kubernetes probes question
    - Contains factual statements on liveness and readiness
    - Explicitly cites the source document chunks
    """
    query = "What is the difference between Kubernetes liveness and readiness probes?"
    initial_state: GraphState = {
        "question": query,
        "original_question": query,
        "documents": [],
        "generation": "",
        "relevance_score": "no",
        "rewrite_count": 0,
        "rewritten_query": None,
        "route_history": [],
        "warning": None
    }

    final_state = crag_graph.invoke(initial_state)
    answer = final_state["generation"]

    assert len(answer) > 50
    # Must contain citations or references to source metadata
    assert "[" in answer and "]" in answer
    # Must mention liveness and readiness concepts accurately
    assert "liveness" in answer.lower()
    assert "readiness" in answer.lower()

def test_fastapi_endpoints():
    """Verify FastAPI /health, /ingest, /query, and Interactive Web UI endpoints."""
    # Web UI endpoints
    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "CRAG" in ui_resp.text
    assert "Agentic State Machine Flow" in ui_resp.text

    ui2_resp = client.get("/ui")
    assert ui2_resp.status_code == 200

    # Health Check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["total_documents_indexed"] > 0

    # Ingest Sample Endpoint
    sample_resp = client.post("/api/ingest-sample", json={"sample_file": "kubernetes_pod_lifecycle.md"})
    assert sample_resp.status_code == 200
    assert sample_resp.json()["status"] == "success"

    # Query Endpoint
    query_payload = {"query": "Explain Kubernetes container startup probe"}
    query_resp = client.post("/query", json=query_payload)
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert "sources" in query_data
    assert query_data["latency_ms"] > 0
    assert isinstance(query_data["sources"], list)
