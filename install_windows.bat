@echo off
rem Erstellt eine virtuelle Python-Umgebung und installiert alle benoetigten Pakete.
rem PyTorch wird mit CUDA 12.1 installiert (benoetigt kompatible NVIDIA-GPU und Treiber).

echo ===================================================
echo Starte Installation fuer MATA-2026 Projekt (Windows)
echo ===================================================

echo.
echo [1/3] Erstelle virtuelle Umgebung (.venv)...
python -m venv .venv

echo.
echo [2/3] Aktiviere Umgebung und aktualisiere pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/3] Installiere Module aus requirements.txt (inkl. PyTorch CUDA 12.1)...
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo.
echo ===================================================
echo Installation erfolgreich!
echo Umgebung aktivieren mit: .venv\Scripts\activate
echo Gesamte Pipeline ausfuehren mit: python run_pipeline.py
echo ===================================================
pause
