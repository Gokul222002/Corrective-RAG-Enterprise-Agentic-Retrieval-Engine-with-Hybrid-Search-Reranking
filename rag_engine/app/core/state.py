from typing import List, Optional, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    State schema for the Corrective RAG (CRAG) Agentic cycle.
    
    Attributes:
        question: User query (original or reformulated)
        original_question: The initial question submitted by user
        documents: List of retrieved & reranked candidate Document chunks
        generation: Grounded LLM generated answer
        relevance_score: Binary grading outcome ('yes' or 'no')
        rewrite_count: Guardrail counter for query rewriting (max 2)
        rewritten_query: Most recent query rewrite, if applicable
        route_history: Audit trail of nodes executed in the cycle
        warning: Warning message if fallback occurred due to max rewrites
    """
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    relevance_score: str
    rewrite_count: int
    rewritten_query: Optional[str]
    route_history: List[str]
    warning: Optional[str]
