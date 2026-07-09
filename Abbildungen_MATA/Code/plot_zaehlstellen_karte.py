"""
Dieses Skript erzeugt eine schlichte Karte des Kantons Schwyz und zeichnet die
fuenf verwendeten ASTRA-Zaehlstellen als beschriftete Punkte ein. Die Karte dient
als Abbildung fuer die Maturaarbeit und wird hochaufloesend (PNG) sowie als
Vektorgrafik (PDF) fuer den Druck gespeichert.

Die Kantonsgrenze wird aus einer offenen Geodatenquelle (swissBOUNDARIES3D von
swisstopo bzw. ein daraus abgeleitetes Kantons-GeoJSON/TopoJSON) unter data/geo/
gelesen. Fehlt diese Datei, gibt das Skript einen klaren Downloadhinweis aus,
statt abzustuerzen.

Eigenstaendig ausfuehrbar mit:
    python Abbildungen_MATA/Code/plot_zaehlstellen_karte.py
"""

import os
import sys

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from shapely.geometry import Point

# ---------------------------------------------------------
# Pfade (relativ zum Projektroot, unabhaengig vom Arbeitsverzeichnis)
# ---------------------------------------------------------
# Skript liegt in Abbildungen_MATA/Code/ -> zwei Ebenen hoch ist der Projektroot.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

GEO_DIR = os.path.join(PROJECT_ROOT, "data", "geo")
# Moegliche Dateinamen der Kantonsgrenze (erste vorhandene Datei wird verwendet).
BOUNDARY_CANDIDATES = [
    "swiss_cantons.topojson",
    "swiss_cantons.geojson",
    "swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp",
    "swissboundaries3d_kantonsgebiet.gpkg",
]

SAVE_DIR = os.path.join(PROJECT_ROOT, "Abbildungen_MATA", "Abbildungen")
SAVE_PATH_PNG = os.path.join(SAVE_DIR, "zaehlstellen_karte.png")
SAVE_PATH_PDF = os.path.join(SAVE_DIR, "zaehlstellen_karte.pdf")

# ---------------------------------------------------------
# Koordinatensysteme
# ---------------------------------------------------------
CRS_LV95 = "EPSG:2056"   # Schweizer Landeskoordinaten (Zielsystem, passt zu den Punkten)
CRS_WGS84 = "EPSG:4326"  # Fallback fuer Quellen ohne hinterlegtes CRS (Laengen-/Breitengrad)

# ---------------------------------------------------------
# Zaehlstellen (Name, Ost E, Nord N in LV95 / EPSG:2056)
# ---------------------------------------------------------
STATIONS = [
    {"name": "Brunnen, Mositunnel", "east": 2689129, "north": 1205058},
    {"name": "Sattel",              "east": 2691106, "north": 1215398},
    {"name": "Wangen SZ",           "east": 2710351, "north": 1227960},
    {"name": "Wollerau, Blatttunnel", "east": 2695600, "north": 1228200},
    {"name": "Schwyz",              "east": 2690230, "north": 1207580},
]

# Beschriftungs-Versatz je Station (dx, dy in Metern) sowie Textausrichtung.
# Manuell gesetzt, damit sich die Labels nicht ueberlappen und gut lesbar bleiben.
LABEL_OFFSETS = {
    "Brunnen, Mositunnel":   {"dx": 0,     "dy": -1100, "ha": "center", "va": "top"},
    "Sattel":                {"dx": 1100,  "dy": 0,     "ha": "left",   "va": "center"},
    "Wangen SZ":             {"dx": -1100, "dy": 0,     "ha": "right",  "va": "center"},
    "Wollerau, Blatttunnel": {"dx": 0,     "dy": 1100,  "ha": "center", "va": "bottom"},
    "Schwyz":                {"dx": 1100,  "dy": 400,   "ha": "left",   "va": "bottom"},
}

# ---------------------------------------------------------
# Stil (Kartengrundlage mit Strassen/Seen/Ortschaften, dezente Overlays)
# ---------------------------------------------------------
FIGURE_SIZE = (8, 9)
COLOR_BG = "white"

# Kartenkachel-Grundlage: zeigt Strassen, Gewaesser (Seen) und groessere Orte.
# CartoDB "Voyager" ist beschriftet und dennoch aufgeraeumt. Alternativen:
#   cx.providers.OpenStreetMap.Mapnik (detailreicher), CartoDB.Positron (heller).
BASEMAP_SOURCE = cx.providers.CartoDB.Voyager
BASEMAP_ZOOM = 11               # Detailgrad der Kacheln (hoeher = mehr Strassen)

COLOR_CANTON_FACE = "none"      # keine Fuellung, damit die Grundkarte sichtbar bleibt
COLOR_CANTON_EDGE = "#c0392b"   # kraeftige Kantonsgrenze, klar von Strassen unterscheidbar
CANTON_LINEWIDTH = 2.0

COLOR_MARKER_FACE = "#1a5fb4"   # kraeftiges Blau, einheitlich, gut sichtbar auf der Karte
COLOR_MARKER_EDGE = "white"
MARKER_SIZE = 90
MARKER_EDGEWIDTH = 1.5

COLOR_LABEL = "#000000"         # reines Schwarz fuer die Beschriftung
FONT_SIZE_LABEL = 11
FONT_WEIGHT_LABEL = "bold"      # fett fuer besseren Kontrast auf der Karte
LABEL_HALO_WIDTH = 4.5          # kraeftiger weisser Umriss hinter dem Text fuer Lesbarkeit

MAP_MARGIN = 2500               # Rand um den Kanton herum in Metern

# Massstabsbalken (dezent, unten links). Bei False wird er weggelassen.
SHOW_SCALEBAR = True
SCALEBAR_LENGTH_M = 5000        # 5 km
SCALEBAR_LABEL = "5 km"
COLOR_SCALEBAR = "#5f6368"


def find_boundary_file():
    """
    Sucht in data/geo/ nach der ersten vorhandenen Grenzdatei aus der
    Kandidatenliste. Gibt den vollen Pfad zurueck oder None, falls keine
    Datei gefunden wird.
    """
    for filename in BOUNDARY_CANDIDATES:
        path = os.path.join(GEO_DIR, filename)
        if os.path.exists(path):
            return path
    return None


def print_missing_boundary_hint():
    """
    Gibt einen klaren Hinweis aus, wie die Kantonsgrenze beschafft und abgelegt
    werden muss, falls unter data/geo/ keine passende Datei gefunden wurde.
    """
    print("FEHLER: Es wurde keine Kantonsgrenze in data/geo/ gefunden.")
    print("")
    print("Bitte eine offene Geodatenquelle herunterladen und unter")
    print(f"  {GEO_DIR}")
    print("ablegen. Erwartete Dateinamen (einer genuegt):")
    for filename in BOUNDARY_CANDIDATES:
        print(f"  - {filename}")
    print("")
    print("Bezugsquellen (offen):")
    print("  - swissBOUNDARIES3D (swisstopo):")
    print("    https://www.swisstopo.admin.ch/de/landschaftsmodell-swissboundaries3d")
    print("  - Kantons-TopoJSON (WGS84), z. B. Gist 'cantons.geojson':")
    print("    https://gist.github.com/cmutel/a2e0f2e48278deeedf19846c39cee4da/raw")
    print("    -> als data/geo/swiss_cantons.topojson speichern")


def load_schwyz_boundary(path):
    """
    Laedt die Grenzdatei, waehlt den Kanton Schwyz aus und reprojiziert die
    Geometrie bei Bedarf auf EPSG:2056, sodass sie zu den Zaehlstellen passt.
    """
    gdf = gpd.read_file(path)

    # Quellen ohne hinterlegtes CRS (typisch bei TopoJSON) sind in WGS84 kodiert.
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)

    schwyz = select_schwyz(gdf)

    # Auf Schweizer Landeskoordinaten reprojizieren (gleiches System wie die Punkte).
    schwyz = schwyz.to_crs(CRS_LV95)
    return schwyz


def select_schwyz(gdf):
    """
    Filtert aus einem GeoDataFrame den Kanton Schwyz heraus. Robust gegenueber
    unterschiedlichen Attributnamen: zuerst ueber das Kuerzel 'SZ', andernfalls
    ueber ein Namensfeld, das 'Schwyz' enthaelt.
    """
    # 1) Kuerzel-Spalte 'id' (z. B. im TopoJSON: id == 'SZ')
    if "id" in gdf.columns:
        mask = gdf["id"].astype(str).str.upper() == "SZ"
        if mask.any():
            return gdf[mask]

    # 2) Beliebige Textspalte, die den Namen 'Schwyz' enthaelt
    for column in gdf.columns:
        if column == gdf.geometry.name:
            continue
        try:
            mask = gdf[column].astype(str).str.contains("Schwyz", case=False, na=False)
        except (AttributeError, TypeError):
            continue
        if mask.any():
            return gdf[mask]

    # 3) Enthaelt die Datei nur ein Objekt, wird dieses verwendet.
    if len(gdf) == 1:
        return gdf

    raise ValueError(
        "Kanton Schwyz konnte in der Grenzdatei nicht eindeutig identifiziert werden."
    )


def build_stations_gdf():
    """
    Erstellt ein GeoDataFrame der Zaehlstellen aus den LV95-Koordinaten.
    """
    names = [s["name"] for s in STATIONS]
    geometry = [Point(s["east"], s["north"]) for s in STATIONS]
    return gpd.GeoDataFrame({"name": names}, geometry=geometry, crs=CRS_LV95)


def draw_scalebar(ax):
    """
    Zeichnet einen dezenten Massstabsbalken unten links in die Karte. Da die
    Koordinaten in Metern (LV95) vorliegen, entspricht die Balkenlaenge direkt
    einer Distanz in Metern.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Startpunkt leicht vom linken/unteren Rand eingerueckt.
    x_start = x_min + (x_max - x_min) * 0.06
    y_pos = y_min + (y_max - y_min) * 0.05
    x_end = x_start + SCALEBAR_LENGTH_M

    ax.plot([x_start, x_end], [y_pos, y_pos],
            color=COLOR_SCALEBAR, lw=2.0, solid_capstyle="butt", zorder=5)
    ax.text((x_start + x_end) / 2.0, y_pos + (y_max - y_min) * 0.012,
            SCALEBAR_LABEL, ha="center", va="bottom",
            fontsize=FONT_SIZE_LABEL - 2, color=COLOR_SCALEBAR, zorder=5)


def plot_map(schwyz, stations):
    """
    Zeichnet die Karte: Kantonsumriss, Zaehlstellen-Punkte und Beschriftungen,
    und speichert das Ergebnis als PNG und PDF.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Kartenausschnitt zuerst festlegen (die Grundkarte richtet sich danach)
    x_min, y_min, x_max, y_max = schwyz.total_bounds
    ax.set_xlim(x_min - MAP_MARGIN, x_max + MAP_MARGIN)
    ax.set_ylim(y_min - MAP_MARGIN, y_max + MAP_MARGIN)
    ax.set_aspect("equal")

    # Kartenkachel-Grundlage laden (Strassen, Seen, Ortschaften) und auf das
    # Datenkoordinatensystem (LV95) projizieren, damit alles deckungsgleich ist.
    cx.add_basemap(ax, crs=schwyz.crs, source=BASEMAP_SOURCE,
                   zoom=BASEMAP_ZOOM, attribution_size=6, zorder=0)

    # Kantonsgrenze als klare Linie ueber der Grundkarte (ohne Fuellung)
    schwyz.plot(ax=ax, facecolor=COLOR_CANTON_FACE, edgecolor=COLOR_CANTON_EDGE,
                linewidth=CANTON_LINEWIDTH, zorder=2)

    # Zaehlstellen als einheitliche Marker
    stations.plot(ax=ax, marker="o", color=COLOR_MARKER_FACE,
                  edgecolor=COLOR_MARKER_EDGE, linewidth=MARKER_EDGEWIDTH,
                  markersize=MARKER_SIZE, zorder=3)

    # Namensbeschriftung direkt neben jedem Punkt (mit weissem Halo fuer Lesbarkeit)
    for _, row in stations.iterrows():
        offset = LABEL_OFFSETS[row["name"]]
        text = ax.text(
            row.geometry.x + offset["dx"],
            row.geometry.y + offset["dy"],
            row["name"],
            ha=offset["ha"], va=offset["va"],
            fontsize=FONT_SIZE_LABEL, color=COLOR_LABEL,
            fontweight=FONT_WEIGHT_LABEL, zorder=4,
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=LABEL_HALO_WIDTH, foreground="white"),
            path_effects.Normal(),
        ])

    # Grenzen erneut setzen (add_basemap kann den Ausschnitt leicht veraendern)
    ax.set_xlim(x_min - MAP_MARGIN, x_max + MAP_MARGIN)
    ax.set_ylim(y_min - MAP_MARGIN, y_max + MAP_MARGIN)

    # Aufgeraeumte Achsen
    ax.set_axis_off()

    if SHOW_SCALEBAR:
        draw_scalebar(ax)

    # Hochaufloesend (PNG) und als Vektorgrafik (PDF) speichern
    fig.savefig(SAVE_PATH_PNG, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    fig.savefig(SAVE_PATH_PDF, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)

    print("Karte gespeichert unter:")
    print(f"  {SAVE_PATH_PNG}")
    print(f"  {SAVE_PATH_PDF}")


def main():
    """
    Einstiegspunkt: Grenzdatei suchen, Kanton Schwyz laden und Karte zeichnen.
    """
    boundary_path = find_boundary_file()
    if boundary_path is None:
        print_missing_boundary_hint()
        sys.exit(1)

    schwyz = load_schwyz_boundary(boundary_path)
    stations = build_stations_gdf()
    plot_map(schwyz, stations)


if __name__ == "__main__":
    main()
