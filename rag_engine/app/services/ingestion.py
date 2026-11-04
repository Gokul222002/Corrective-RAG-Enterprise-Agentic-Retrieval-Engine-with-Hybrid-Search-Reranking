import io
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from rag_engine.app.config import settings
from rag_engine.app.services.retriever import hybrid_retriever

class IngestionPipeline:
    """
    Enterprise Ingestion Service:
    - Recursive & layout-aware chunking (chunk_size=600, chunk_overlap=100)
    - Multimodal format extraction (.txt, .md, .pdf, raw string)
    - Metadata enrichment (source, chunk_id, index)
    - Dual synchronization into Qdrant & BM25 indices
    """

    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "### ", "## ", "# ", ". ", " ", ""],
            keep_separator=True
        )

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Extracts text content from .txt, .md, or .pdf byte buffers."""
        ext = Path(filename).suffix.lower()
        if ext in [".txt", ".md", ".markdown", ".log", ".json", ".csv"]:
            return file_bytes.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append(f"<!-- Page {page_idx + 1} -->\n{text}")
            return "\n\n".join(pages)
        else:
            # Fallback to text decoding
            return file_bytes.decode("utf-8", errors="replace")

    def extract_text_from_path(self, file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        with open(path, "rb") as f:
            return self.extract_text_from_bytes(f.read(), path.name)

    def process_document(
        self,
        content: str,
        source: str = "document",
        extra_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Splits text content into overlapping chunks with structured metadata.
        """
        if not content or not content.strip():
            return []

        extra_metadata = extra_metadata or {}
        raw_chunks = self.splitter.split_text(content)
        
        doc_chunks: List[Document] = []
        for idx, text_chunk in enumerate(raw_chunks):
            # Compute deterministic chunk hash
            chunk_hash = hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()[:8]
            clean_source = Path(source).name
            chunk_id = f"{clean_source}#chunk_{idx:03d}_{chunk_hash}"
            
            meta = {
                "source": clean_source,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "char_length": len(text_chunk),
                **extra_metadata
            }
            doc_chunks.append(Document(page_content=text_chunk, metadata=meta))

        # Index into hybrid retriever (Qdrant + BM25)
        hybrid_retriever.index_documents(doc_chunks)
        return doc_chunks

    def ingest_file(self, file_path: Union[str, Path], extra_metadata: Dict[str, Any] = None) -> List[Document]:
        """Ingests file from local filesystem."""
        path = Path(file_path)
        content = self.extract_text_from_bytes(path.read_bytes(), path.name)
        return self.process_document(content, source=path.name, extra_metadata=extra_metadata)

ingestion_pipeline = IngestionPipeline()
