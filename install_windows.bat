@echo off
rem "Dieses Skript erstellt eine virtuelle Python-Umgebung (.venv) und installiert alle für das Projekt benötigten Pakete über pip."

echo ===================================================
echo Starte Installation für MATA-2026 Projekt (Windows)...
echo ===================================================

echo.
echo [1/3] Erstelle virtuelle Umgebung (.venv)...
python -m venv .venv

echo.
echo [2/3] Aktiviere Umgebung und aktualisiere pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/3] Installiere Module aus requirements.txt...
pip install -r requirements.txt

echo.
echo ===================================================
echo Installation erfolgreich! 
echo Um das Projekt auszufuehren, nutze die aktivierte Umgebung.
echo (Der Befehl dazu lautet: .venv\Scripts\activate)
echo ===================================================
pause
