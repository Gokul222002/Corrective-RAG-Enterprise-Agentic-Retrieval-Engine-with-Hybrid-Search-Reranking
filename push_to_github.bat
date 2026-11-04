@echo off
setlocal
echo ===================================================
echo  Push Enterprise CRAG System to GitHub
echo ===================================================
echo.
set /p GITHUB_TOKEN="Enter your GitHub Personal Access Token (PAT): "
if "%GITHUB_TOKEN%"=="" (
    echo Error: Token cannot be empty.
    pause
    exit /b 1
)

echo.
echo Pushing to https://github.com/Gokul222002/Corrective-RAG-Enterprise-Agentic-Retrieval-Engine-with-Hybrid-Search-Reranking...
"%~dp0.venv\Scripts\python.exe" "%~dp0push_github.py" "%GITHUB_TOKEN%"
echo.
pause
