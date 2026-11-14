@echo off
rem Visor de resultats: arrenca el servidor dels CSV de results/ i obre el navegador.
rem Es pot executar fent doble clic o des de qualsevol directori.
set ROOT=%~dp0datathon-UPC-maths-qualcomm-arduino-unoq-main\datathon-UPC-maths-qualcomm-arduino-unoq-main
start "Visor de resultats" /min cmd /k python %ROOT%\results-viewer\server.py
timeout /t 1 /nobreak >nul
start "" http://127.0.0.1:8001
