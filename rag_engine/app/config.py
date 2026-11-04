import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & Providers
    GOOGLE_API_KEY: str = Field(default="", description="Google Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Gemini model for generation & reasoning")
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004", description="Gemini embedding model")
    
    # Storage & Indices
    QDRANT_LOCATION: str = Field(default=str(BASE_DIR / "qdrant_storage"), description="Local directory path or :memory:")
    QDRANT_COLLECTION: str = Field(default="enterprise_crag_corpus", description="Vector collection name")
    
    # Ingestion & Chunking parameters
    CHUNK_SIZE: int = Field(default=600, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=100, description="Chunk overlap in characters")
    
    # Retrieval & Reranker parameters
    TOP_K_RETRIEVAL: int = Field(default=8, description="Initial candidates per retriever (Dense & BM25)")
    TOP_N_RERANK: int = Field(default=3, description="Final top reranked chunks for generation context")
    RERANKER_MODEL: str = Field(default="ms-marco-TinyBERT-L-2-v2", description="FlashRank or Cross-Encoder model")
    
    # Agent Guardrails
    MAX_REWRITES: int = Field(default=2, description="Maximum query rewriting iterations")
    
    # Development / Fallback mode
    MOCK_MODE: bool = Field(default=False, description="Force deterministic mock mode for offline testing")

settings = Settings()
