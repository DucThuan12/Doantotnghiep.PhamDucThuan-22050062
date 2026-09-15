@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Khong tim thay Python trong PATH.
    echo Hay cai Python va chon tuy chon Add Python to PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Dang tao moi truong ao .venv...
    python -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :error

pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [PASS] Da cai dependency. Tiep theo chay CHECK_TUAN6_WINDOWS.bat.
pause
exit /b 0

:error
echo.
echo [FAIL] Cai dependency khong thanh cong. Gui dong loi dau tien de debug.
pause
exit /b 1
