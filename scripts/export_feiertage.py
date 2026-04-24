"""
"Lädt die gesetzlichen Feiertage für den Kanton Schwyz (SZ) für die Jahre 2015 bis 2026 herunter und speichert sie als CSV-Datei im Verzeichnis 'data/holidays'."
"""

import holidays
import csv
import os

# Kanton und Jahresbereich definieren
kanton = 'SZ'
jahre = range(2015, 2027)  # 2015 bis 2026 inkl.

# CSV-Datei im data/holiday Ordner erstellen
basis = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ordner = os.path.join(basis, "data", "v1_raw", "holidays")
csv_datei = os.path.join(ordner, f"feiertage_{kanton}_2015_2026.csv")

alle_eintraege = []

for jahr in jahre:
    ch_holidays = holidays.CH(subdiv=kanton, years=jahr)
    for date, name in sorted(ch_holidays.items()):
        alle_eintraege.append([date, jahr, name])

# In CSV schreiben
with open(csv_datei, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Datum", "Jahr", "Feiertag"])  # Header
    for eintrag in alle_eintraege:
        writer.writerow(eintrag)

print(f"CSV gespeichert: {csv_datei}")
print(f"Anzahl Einträge: {len(alle_eintraege)}")