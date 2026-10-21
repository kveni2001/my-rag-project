@echo off
title NexusRAG - FastAPI Backend Server
echo ===================================================
echo   Starting NexusRAG FastAPI Backend on Port 8000
echo ===================================================
cd /d "%~dp0rag_pipeline"
call ".\venv\Scripts\activate.bat"
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
pause

