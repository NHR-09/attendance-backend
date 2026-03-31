@echo off
echo Starting Native Attendance Backend...
cd /d "%~dp0"
call venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
