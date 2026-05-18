# Modell-Gewichte

Hier liegen die exportierten MLP- und Linear-Regression-Gewichte pro Station.

Dateinamen-Schema:

- `{station_id}_mlp.json`
- `{station_id}_linear.json`

Aktuelle Dateien sind Platzhalter, erzeugt von `webapp/scripts/generate_placeholders.mjs`.
Sobald die echten Modelle trainiert sind, überschreiben die Python-Export-Snippets aus
`webapp/README.md` diese Dateien.
