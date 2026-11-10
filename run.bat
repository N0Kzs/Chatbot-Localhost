@echo off
REM J.A.R.V.I.S. launcher — creates venv on first run, then starts the server.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo [JARVIS] Creating virtual environment...
  python -m venv venv || goto :err
  venv\Scripts\python -m pip install --upgrade pip
  venv\Scripts\python -m pip install -r requirements.txt || goto :err
)
start "" http://localhost:8000
venv\Scripts\python main.py
goto :eof
:err
echo [JARVIS] Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
