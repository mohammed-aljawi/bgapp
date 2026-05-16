@echo off
cd /d "%~dp0"
echo Starting NewBG...
echo.
echo If this is your first time, run:
echo pip install -r requirements.txt
echo.
streamlit run app.py --server.port 8510 --server.fileWatcherType none
pause
