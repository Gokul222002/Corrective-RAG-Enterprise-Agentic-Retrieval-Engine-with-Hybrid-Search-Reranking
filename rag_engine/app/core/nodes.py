from typing import Dict, Any, List
from rag_engine.app.core.state import GraphState
from rag_engine.app.services.retriever import hybrid_retriever
from rag_engine.app.services.llm import llm_service
from rag_engine.app.config import settings

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve candidate chunks using hybrid dense (Qdrant) + sparse (BM25) search,
    deduplicate, and rerank to produce top-3 chunks.
    """
    query = state.get("question", "")
    route_history = list(state.get("route_history", []))
    route_history.append("retrieve")

    docs = hybrid_retriever.hybrid_retrieve_and_rerank(
        query=query,
        top_k=settings.TOP_K_RETRIEVAL,
        top_n=settings.TOP_N_RERANK
    )

    return {
        "documents": docs,
        "route_history": route_history
    }

def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Grades whether the retrieved context contains relevant information for the user question.
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    route_history = list(state.get("route_history", []))
    route_history.append("grade_documents")

    grade_result = llm_service.grade_relevance(question=question, documents=documents)
    
    return {
        "relevance_score": grade_result.binary_score,
        "route_history": route_history
    }

def transform_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Re-writes and expands search query based on semantic intent,
    increments rewrite_count, and prepares loop back to retrieve_node.
    """
    current_question = state.get("question", "")
    documents = state.get("documents", [])
    current_rewrites = state.get("rewrite_count", 0)
    route_history = list(state.get("route_history", []))
    route_history.append("transform_query")

    rewrite_res = llm_service.rewrite_query(question=current_question, current_documents=documents)
    new_count = current_rewrites + 1

    return {
        "question": rewrite_res.improved_query,
        "rewritten_query": rewrite_res.improved_query,
        "rewrite_count": new_count,
        "route_history": route_history
    }

def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    Grounded generation prompt that answers the query and explicitly cites document chunks.
    Includes warning flag if maximum rewrites occurred with weak context.
    """
    original_question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents", [])
    relevance_score = state.get("relevance_score", "no")
    rewrite_count = state.get("rewrite_count", 0)
    route_history = list(state.get("route_history", []))
    route_history.append("generate")

    warning = state.get("warning")
    if rewrite_count >= settings.MAX_REWRITES and relevance_score == "no":
        warning = f"Insufficient context available in knowledge base after {settings.MAX_REWRITES} search reformulation attempts."

    answer = llm_service.generate_answer(
        question=original_question,
        documents=documents,
        warning=warning
    )

    return {
        "generation": answer,
        "warning": warning,
        "route_history": route_history
    }

def decide_next_step(state: GraphState) -> str:
    """
    Conditional router edge:
    - If context is graded 'yes' -> 'generate'
    - If graded 'no' and rewrite_count < 2 -> 'transform_query'
    - If rewrite_count >= 2 -> 'generate' (with insufficient context warning)
    """
    relevance = state.get("relevance_score", "no")
    rewrite_count = state.get("rewrite_count", 0)

    if relevance == "yes":
        return "generate"
    elif rewrite_count < settings.MAX_REWRITES:
        return "transform_query"
    else:
        return "generate"
