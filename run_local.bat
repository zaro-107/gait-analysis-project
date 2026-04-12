@echo off
echo ================================
echo   Running Project on Localhost
echo ================================

REM Activate virtual environment
IF EXIST venv (
    call venv\Scripts\activate
)

REM Install dependencies (optional)
pip install -r requirements.txt

echo Starting Backend...
start cmd /k "cd backend && python app.py"

echo Starting Frontend...
start cmd /k "cd frontend && npm start"

echo ================================
echo Backend: http://127.0.0.1:5000
echo Frontend: http://localhost:3000
echo ================================
pause