@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo FITMOTION AI - CAI GIỌNG NOI TIENG VIET PIPER
ECHO ==================================================
python scripts\setup_piper_vi.py
if errorlevel 1 (
  echo.
  echo Cai Piper that bai. Kiem tra mang va moi truong Python roi chay lai.
  pause
  exit /b 1
)
echo.
echo Da cai xong. Hay mo lai FitMotion AI va thu tao tieu chi am thanh trong Admin.
pause
