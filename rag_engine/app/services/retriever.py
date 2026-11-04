import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import qdrant_client
from qdrant_client.http import models as qmodels

from rag_engine.app.config import settings
from rag_engine.app.services.llm import get_embeddings

class HybridRetrieverService:
    """
    Enterprise Hybrid Search engine combining:
    1. Dense Vector Search via Qdrant (local on-disk or in-memory)
    2. Sparse Keyword Search via BM25
    3. Neural Cross-Encoder Reranking via FlashRank / Lightweight ONNX
    """

    def __init__(self):
        self.embeddings = get_embeddings()
        self.collection_name = settings.QDRANT_COLLECTION
        
        # Initialize Qdrant Client (local directory or in-memory)
        if settings.QDRANT_LOCATION == ":memory:":
            self.client = qdrant_client.QdrantClient(location=":memory:")
        else:
            storage_path = Path(settings.QDRANT_LOCATION)
            storage_path.mkdir(parents=True, exist_ok=True)
            try:
                self.client = qdrant_client.QdrantClient(path=str(storage_path))
            except Exception as e:
                print(f"[Info] Storage folder locked by active server ({e}). Using isolated memory client.")
                self.client = qdrant_client.QdrantClient(location=":memory:")
            
        self._ensure_collection()

        # In-memory document store & BM25 index
        self.doc_store: Dict[str, Document] = {}
        self.bm25_corpus: List[List[str]] = []
        self.bm25_doc_ids: List[str] = []
        self.bm25_index: Optional[BM25Okapi] = None
        
        # Load any existing persisted points from Qdrant into BM25
        self._load_persisted_corpus()
        
        # FlashRank ranker initialization
        self.ranker = None
        self._init_ranker()

    def _ensure_collection(self):
        """Creates Qdrant collection if not already present."""
        test_embed = self.embeddings.embed_query("test dimension probe")
        vector_dim = len(test_embed)

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_dim,
                    distance=qmodels.Distance.COSINE
                )
            )

    def _load_persisted_corpus(self):
        """Loads points from Qdrant into memory doc_store and initializes BM25 index on startup."""
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )
            for record in records:
                payload = record.payload or {}
                content = payload.get("content", "")
                chunk_id = payload.get("chunk_id", str(record.id))
                metadata = {k: v for k, v in payload.items() if k != "content"}
                self.doc_store[chunk_id] = Document(page_content=content, metadata=metadata)
            if self.doc_store:
                self._rebuild_bm25()
        except Exception as e:
            print(f"[Info] Cold start corpus loading: {e}")

    def _init_ranker(self):
        """Initializes FlashRank cross-encoder reranker with graceful fallback."""
        try:
            from flashrank import Ranker
            self.ranker = Ranker(
                model_name="ms-marco-TinyBERT-L-2-v2",
                cache_dir=str(Path(settings.QDRANT_LOCATION) / "flashrank")
            )
        except Exception as e:
            print(f"[Info] FlashRank initialized in fallback mode: {e}")
            self.ranker = None

    def index_documents(self, documents: List[Document]):
        """
        Indexes chunks into both Qdrant Vector Store and the BM25 index.
        """
        if not documents:
            return

        texts = [d.page_content for d in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        points = []
        for i, (doc, vec) in enumerate(zip(documents, embeddings)):
            chunk_id = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc.metadata["chunk_id"] = chunk_id
            
            # Save into memory doc store
            self.doc_store[chunk_id] = doc
            
            # Use deterministic integer id for Qdrant point_id
            numeric_id = int(hashlib.md5(chunk_id.encode("utf-8")).hexdigest()[:15], 16)
            
            payload = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": chunk_id,
                **doc.metadata
            }
            points.append(qmodels.PointStruct(
                id=numeric_id,
                vector=vec,
                payload=payload
            ))

        # Batch upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        # Rebuild BM25 index
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Rebuilds the BM25 index from all indexed documents."""
        self.bm25_doc_ids = list(self.doc_store.keys())
        self.bm25_corpus = [
            self.doc_store[doc_id].page_content.lower().split()
            for doc_id in self.bm25_doc_ids
        ]
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)

    def dense_search(self, query: str, top_k: int = 8) -> List[Tuple[Document, float]]:
        """Dense semantic search via Qdrant vector cosine similarity."""
        query_vec = self.embeddings.embed_query(query)
        try:
            hits = []
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_k
                )
                hits = response.points
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=top_k
                )
                
            results = []
            for hit in hits:
                payload = hit.payload or {}
                content = payload.get("content", "")
                metadata = {k: v for k, v in payload.items() if k != "content"}
                score = float(hit.score) if getattr(hit, "score", None) is not None else 0.0
                metadata["dense_score"] = score
                doc = Document(page_content=content, metadata=metadata)
                results.append((doc, score))
            return results
        except Exception as e:
            print(f"[Warning] Qdrant search error: {e}")
            return []

    def sparse_search(self, query: str, top_k: int = 8) -> List[Tuple[Document, float]]:
        """Sparse lexical search via BM25."""
        if not self.bm25_index or not self.bm25_doc_ids:
            return []
        tokens = query.lower().split()
        if not tokens:
            return []
            
        scores = self.bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                doc_id = self.bm25_doc_ids[idx]
                orig_doc = self.doc_store[doc_id]
                metadata = dict(orig_doc.metadata)
                metadata["sparse_score"] = score
                doc = Document(page_content=orig_doc.page_content, metadata=metadata)
                results.append((doc, score))
        return results

    def hybrid_retrieve_and_rerank(self, query: str, top_k: int = 8, top_n: int = 3) -> List[Document]:
        """
        1. Query both Qdrant and BM25
        2. Deduplicate candidates by chunk ID
        3. Run candidates through Cross-Encoder/FlashRank reranker to produce top-3 chunks
        """
        dense_results = self.dense_search(query, top_k=top_k)
        sparse_results = self.sparse_search(query, top_k=top_k)

        candidates: Dict[str, Document] = {}

        for doc, score in dense_results:
            c_id = doc.metadata.get("chunk_id")
            if c_id:
                candidates[c_id] = doc

        for doc, score in sparse_results:
            c_id = doc.metadata.get("chunk_id")
            if c_id:
                if c_id in candidates:
                    candidates[c_id].metadata["sparse_score"] = score
                else:
                    candidates[c_id] = doc

        candidate_docs = list(candidates.values())
        if not candidate_docs:
            return []

        # Rerank candidates
        reranked_docs = self._rerank(query, candidate_docs, top_n=top_n)
        return reranked_docs

    def _rerank(self, query: str, candidates: List[Document], top_n: int = 3) -> List[Document]:
        """Applies FlashRank or heuristic cross-entropy scoring."""
        if self.ranker is not None:
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": d.metadata.get("chunk_id", str(i)), "text": d.page_content}
                    for i, d in enumerate(candidates)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                ranked_passages = self.ranker.rerank(rerank_req)
                
                # Match back to Document objects
                ranked_map = {str(p["id"]): p.get("score", 0.0) for p in ranked_passages}
                
                for doc in candidates:
                    cid = doc.metadata.get("chunk_id")
                    doc.metadata["rerank_score"] = float(ranked_map.get(cid, 0.0))
                
                sorted_docs = sorted(
                    candidates,
                    key=lambda d: d.metadata.get("rerank_score", 0.0),
                    reverse=True
                )
                return sorted_docs[:top_n]
            except Exception as e:
                print(f"[Warning] FlashRank reranking error ({e}). Using hybrid score fusion.")

        # Heuristic Reciprocal Rank / Score Fusion fallback
        for doc in candidates:
            d_score = doc.metadata.get("dense_score", 0.0) or 0.0
            s_score = doc.metadata.get("sparse_score", 0.0) or 0.0
            fusion_score = (d_score * 0.6) + (min(s_score / 10.0, 1.0) * 0.4)
            doc.metadata["rerank_score"] = round(fusion_score, 4)

        sorted_docs = sorted(
            candidates,
            key=lambda d: d.metadata.get("rerank_score", 0.0),
            reverse=True
        )
        return sorted_docs[:top_n]

    def count(self) -> int:
        """Returns total chunks indexed in Qdrant."""
        try:
            if hasattr(self.client, "count"):
                info = self.client.count(collection_name=self.collection_name)
                return info.count or len(self.doc_store)
            info = self.client.get_collection(self.collection_name)
            return info.points_count or len(self.doc_store)
        except Exception:
            return len(self.doc_store)

# Global singleton
hybrid_retriever = HybridRetrieverService()
