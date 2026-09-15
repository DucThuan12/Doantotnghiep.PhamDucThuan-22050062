@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe scripts\check_schedule_reason1.py
) else (
  python scripts\check_schedule_reason1.py
)
pause
