from langgraph.graph import StateGraph, END
from rag_engine.app.core.state import GraphState
from rag_engine.app.core.nodes import (
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    generate_node,
    decide_next_step
)

def build_crag_graph():
    """
    Assembles the Corrective RAG (CRAG) Agentic StateGraph:
    
       [START]
          │
          ▼
    ┌─────────────┐
    │  retrieve   │◄───────────────────────┐
    └──────┬──────┘                        │
           │                               │
           ▼                               │
    ┌──────────────────┐                   │
    │ grade_documents  │                   │
    └──────┬───────────┘                   │
           │                               │
       [Conditional Router]                │
           │                               │
           ├─ ('no' & rewrites < 2) ──► ┌──────────────────┐
           │                            │ transform_query  │
           │                            └──────────────────┘
           ▼ ('yes' OR rewrites >= 2)
    ┌──────────────┐
    │   generate   │
    └──────┬───────┘
           │
           ▼
         [END]
    """
    workflow = StateGraph(GraphState)

    # 1. Define Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("generate", generate_node)

    # 2. Set Entry Point
    workflow.set_entry_point("retrieve")

    # 3. Add Edges
    workflow.add_edge("retrieve", "grade_documents")
    
    # 4. Add Conditional Routing Edge
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )

    # 5. Loop back from query transformation to retrieval
    workflow.add_edge("transform_query", "retrieve")

    # 6. Exit
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()
    return app

# Singleton compiled graph
crag_graph = build_crag_graph()
