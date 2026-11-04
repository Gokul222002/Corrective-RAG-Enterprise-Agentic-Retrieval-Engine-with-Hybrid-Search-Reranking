import sys
import os
from dulwich import porcelain
from dulwich.repo import Repo

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

token = sys.argv[1].strip() if len(sys.argv) > 1 else os.environ.get("GITHUB_TOKEN", "").strip()

if not token:
    print("Error: No GitHub Personal Access Token provided.")
    print("Generate one with 'repo' scope at: https://github.com/settings/tokens")
    sys.exit(1)

username = "Gokul222002"
repo_name = "Corrective-RAG-Enterprise-Agentic-Retrieval-Engine-with-Hybrid-Search-Reranking"
auth_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"

repo = Repo(PROJECT_DIR)

print(f"Connecting to https://github.com/{username}/{repo_name}...")
print(f"Pushing commit {repo.head().decode('ascii')} on branch 'main'...")

try:
    porcelain.push(repo, auth_url, refspecs=[b"refs/heads/main:refs/heads/main"])
    print("\nSUCCESS: All files successfully pushed to GitHub!")
    print(f"Repository URL: https://github.com/{username}/{repo_name}")
except Exception as e:
    print(f"\nPush failed: {type(e).__name__}: {e}")
    sys.exit(1)
