@echo off
title Enterprise CRAG Engine Server
cd /d "%~dp0"
echo =========================================================
echo Starting Enterprise Corrective RAG (CRAG) Engine...
echo =========================================================
if exist .venv\Scripts\python.exe (
    set PYTHONPATH=%~dp0
    .venv\Scripts\python.exe run_server.py
) else (
    echo Error: .venv virtual environment not found.
    pause
)
