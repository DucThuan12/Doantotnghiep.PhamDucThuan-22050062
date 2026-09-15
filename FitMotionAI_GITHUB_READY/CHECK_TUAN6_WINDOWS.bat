@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

python scripts\preflight_week6.py --camera
set RESULT=%ERRORLEVEL%

echo.
if %RESULT% EQU 0 (
    echo KIEM TRA TUAN 6: PASS
) else (
    echo KIEM TRA TUAN 6: FAIL - xem dong FAIL dau tien o phia tren.
)
pause
exit /b %RESULT%
