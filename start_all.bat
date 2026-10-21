@echo off
title NexusRAG System Launcher
echo =============================================================
echo   Launching NexusRAG (Backend + Frontend + Browser)
echo =============================================================

echo 1. Starting Backend API on http://127.0.0.1:8000 ...
start "NexusRAG Backend" cmd /c "%~dp0start_backend.bat"

echo 2. Waiting 3 seconds for backend initialization ...
timeout /t 3 /nobreak >nul

echo 3. Starting Frontend on http://localhost:5173 ...
start "NexusRAG Frontend" cmd /c "%~dp0start_frontend.bat"

echo 4. Waiting 2 seconds then opening browser ...
timeout /t 2 /nobreak >nul
start http://localhost:5173

echo =============================================================
echo   NexusRAG is now running! Keep the server windows open.
echo =============================================================

