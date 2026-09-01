@echo off
title Project Aegis Pro Trading Terminal
cd /d "%~dp0"

echo ========================================================
echo   PROJECT AEGIS v3.0 - REAL-TIME TRADING TERMINAL
echo ========================================================
echo.

echo Freeing port 8000 and 5173 if busy...
powershell -Command "Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Starting FastAPI Real-Time Backend on http://127.0.0.1:8000 ...
start "Aegis Backend API" "%~dp0.venv\Scripts\python.exe" -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000

echo Waiting for backend initialization...
timeout /t 3 /nobreak >nul

echo Starting React Frontend on http://localhost:5173 ...
start "Aegis React Terminal" cmd /c "cd /d ""%~dp0frontend"" && npm run dev"

echo Opening Trading Terminal in browser...
timeout /t 2 /nobreak >nul
start http://localhost:5173

echo.
echo ========================================================
echo   AEGIS TERMINAL IS RUNNING!
echo   Frontend : http://localhost:5173
echo   Backend  : http://127.0.0.1:8000/docs
echo ========================================================