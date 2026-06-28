# Schulferien-Datensatz Kantone 2015–2026 – Quellen und Methode

## Datei
`schulferien_kantone_2015_2026.csv` – eine Zeile pro Tag (2015-01-01 bis 2026-12-31),
Spalte `Datum` (ISO) plus 26 Kantonsspalten (AG, AI, …, ZH) mit Wert 1 (Schulferien)
oder 0 (kein Ferientag). 4383 Zeilen.

## Definition
1 = an diesem Kalendertag hat die repräsentative Schule des Kantons Ferien
(Volksschule / «Alle Schulen» bzw. der kantonale Hauptkalender). Enthalten sind
Sport-, Frühlings-, Sommer-, Herbst- und Weihnachtsferien sowie einzelne
Brückentage, soweit von der Quelle als schulfrei ausgewiesen.

## Quellen
- Primär: schulferien.org, Jahresübersichten Schweiz (Jahrgänge 2014–2024).
  Das Jahr 2014 dient nur dem Januar-Übertrag der Weihnachtsferien 2014/15.
- Ergänzung: OpenHolidays API (openholidaysapi.org) für Kanton-Jahre, die auf
  schulferien.org noch nicht vollständig publiziert waren. Bei mehreren
  Sprach-/Schulregionen eines Kantons wurde die Vereinigung gebildet (jeder Tag,
  an dem mindestens eine Schulregion Ferien hat, zählt als 1).

OpenHolidays wurde verwendet für: JU 2023; GE und UR 2024; GE, UR, VS, GR 2025;
GE, GL, GR, SO, UR, VD, VS, ZG, ZH 2026.

## Bekannte Einschränkungen
- GR 2026: Frühlings- und Sommerferien sind in keiner Quelle vollständig
  publiziert (Graubünden legt Ferien pro Schulregion fest). Für 2026 sind bei GR
  daher nur Herbst- und Weihnachtsferien erfasst. Vor Verwendung von GR 2026 prüfen.
- VS: ab 2025 als Vereinigung der deutsch-/französischsprachigen Regionen und der
  Tourismusgemeinden – dadurch etwas längere Ferienfenster als bei einer
  Einzelregion.
- Quellenwechsel 2024/2025: für einige Kantone wechselt die Quelle von
  schulferien.org auf OpenHolidays; minimale Abweichungen an den Rändern möglich.

## Reproduzierbarkeit
`build_ferien.py` enthält alle Rohdaten (Datumsbereiche) und erzeugt die CSV neu.
Stichproben gegen die Quellen wurden geprüft (Sommerbeginn/-ende, Jahreswechsel,
Kantone ohne Sportferien).
