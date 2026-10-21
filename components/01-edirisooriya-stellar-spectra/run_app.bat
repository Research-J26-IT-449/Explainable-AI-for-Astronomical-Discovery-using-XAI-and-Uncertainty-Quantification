@echo off
echo Starting Module 1 Stellar Spectra Streamlit Application...
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
