#!/bin/bash
# "Dieses Skript erstellt eine virtuelle Python-Umgebung (.venv) und installiert alle für das Projekt benötigten Pakete über pip."

echo "==================================================="
echo "Starte Installation für MATA-2026 Projekt (Mac/Linux)..."
echo "==================================================="

echo ""
echo "[1/3] Erstelle virtuelle Umgebung (.venv)..."
python3 -m venv .venv

echo ""
echo "[2/3] Aktiviere Umgebung und aktualisiere pip..."
source .venv/bin/activate
python3 -m pip install --upgrade pip

echo ""
echo "[3/3] Installiere Module aus requirements.txt..."
pip install -r requirements.txt

echo ""
echo "==================================================="
echo "Installation erfolgreich! "
echo "Um die Umgebung zu aktivieren, führe aus: source .venv/bin/activate"
echo "==================================================="
