@echo off
REM Rayleigh Duplicate Detector — Windows Launcher
REM Double-click this file to start the app
cd /d "%~dp0"
echo Installing dependencies...
pip install -q streamlit numpy scipy matplotlib pandas 2>nul
echo Starting Rayleigh Duplicate Detector...
streamlit run app.py
pause
