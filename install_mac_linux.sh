#!/bin/bash
# Erstellt eine virtuelle Python-Umgebung und installiert alle benoetigten Pakete.
# PyTorch wird als CPU-Version installiert (CUDA nicht verfuegbar auf Mac/Linux ohne NVIDIA-GPU).

echo "==================================================="
echo "Starte Installation fuer MATA-2026 Projekt (Mac/Linux)"
echo "==================================================="

echo ""
echo "[1/4] Erstelle virtuelle Umgebung (.venv)..."
python3 -m venv .venv

echo ""
echo "[2/4] Aktiviere Umgebung und aktualisiere pip..."
source .venv/bin/activate
python3 -m pip install --upgrade pip

echo ""
echo "[3/4] Installiere PyTorch (CPU-Version)..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[4/4] Installiere restliche Module aus requirements.txt..."
grep -Ev "^(torch|#|$)" requirements.txt | pip install -r /dev/stdin

echo ""
echo "==================================================="
echo "Installation erfolgreich!"
echo "Umgebung aktivieren mit: source .venv/bin/activate"
echo "Gesamte Pipeline ausfuehren mit: python run_pipeline.py"
echo "==================================================="
