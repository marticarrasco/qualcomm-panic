@echo off
rem Serve local evaluation CSVs and open the browser.
start "MathQ Results Viewer" /D "%~dp0." /min cmd /k python -X utf8 results-viewer\server.py
timeout /t 1 /nobreak >nul
start "" http://127.0.0.1:8001
