import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add current workspace directory to sys.path and environment
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.environ["PYTHONPATH"] = str(BASE_DIR)

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("[*] Enterprise Corrective RAG (CRAG) Agentic Engine")
    print("=" * 65)
    print(" -> Interactive Web UI & Dashboard: http://localhost:8000")
    print(" -> Local IP Direct Access:         http://127.0.0.1:8000")
    print(" -> OpenAPI / Swagger API Docs:     http://localhost:8000/docs")
    print("=" * 65 + "\n")

    port = int(os.environ.get("PORT", 8000))
    print(f" -> Active Port:                     {port}")
    print("=" * 65 + "\n")

    uvicorn.run(
        "rag_engine.app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
