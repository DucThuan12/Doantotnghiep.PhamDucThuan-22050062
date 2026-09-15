@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

python scripts\preflight_week6.py
if errorlevel 1 (
    echo.
    echo Khong khoi dong vi preflight dang FAIL.
    echo Chay CHECK_TUAN6_WINDOWS.bat de xem day du, sau do sua loi dau tien.
    pause
    exit /b 1
)

start "" http://127.0.0.1:5000
python app.py
pause
