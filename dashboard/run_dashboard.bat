@echo off
setlocal
cd /d "%~dp0"
set "PYCMD="
where py >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD where python >nul 2>nul && set "PYCMD=python"
if not defined PYCMD (
  echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYCMD% -m venv .venv
  if errorlevel 1 goto :fail
)
if not exist ".venv\.deps_ok" (
  echo Installing dependencies ^(first run only, needs internet^)...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  echo ok> ".venv\.deps_ok"
)
echo.
echo Starting IoTGuard dashboard at http://localhost:8501  (Ctrl+C to stop)
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof
:fail
echo.
echo Setup failed - see the messages above and the Troubleshooting section of README.md.
pause
exit /b 1
