"""
Dieses Skript zeichnet eine aufgeraeumte, thesenreife Zeitreihe der ersten zwei
zusammenhaengenden Wochen des Testsets fuer die Station Schwyz R1 (lineare
Regression v8). Es liest die bereits erzeugten Vorhersagen aus
results/model_results/linear_regression_v8/predictions/ und stellt den
gemessenen Verkehr (schwarz) der Vorhersage (orange) gegenueber.

"Zusammenhaengend" bedeutet: das erste lueckenlose Fenster von 336 Stunden
(2 Wochen) mit echten 1h-Schritten, damit die Kurve keine Zeitspruenge enthaelt.

Ergebnis:
    results/model_results/linear_regression_v8/summary/timeseries_clean_Schwyz_R1.png
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Konfiguration ────────────────────────────────────────────────────────────
PRED_CSV = Path("results/model_results/linear_regression_v8/predictions/predictions_720_Schwyz_R1_v8.csv")
OUT_PATH = Path("results/model_results/linear_regression_v8/summary/timeseries_clean_Schwyz_R1.png")

HOURS_PER_WEEK = 7 * 24
N_WEEKS        = 2
WINDOW_HOURS   = N_WEEKS * HOURS_PER_WEEK   # 336 Stunden = 2 Wochen

ACTUAL_COLOR = "black"       # Messwert
PRED_COLOR   = "#DD8452"     # Vorhersage (orange)

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Lineare Regression: Prognose und Messwerte (Schwyz R1, zwei Wochen)"


def first_contiguous_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Gibt die ersten `window` Stunden ohne Zeitluecke (1h-Schritte) zurueck."""
    times = df.index
    start = 0
    run = 1
    for i in range(1, len(times)):
        if times[i] - times[i - 1] == pd.Timedelta(hours=1):
            run += 1
        else:
            # Zeitluecke: neuer zusammenhaengender Abschnitt beginnt bei i
            start = i
            run = 1
        if run >= window:
            return df.iloc[start:start + window]
    # Kein volles Fenster gefunden: die verfuegbaren Daten ab Beginn nehmen
    return df.iloc[:window]


def main():
    """Liest die Vorhersagen und speichert die aufgeraeumte Zeitreihe."""
    df = pd.read_csv(PRED_CSV, parse_dates=["datetime"]).set_index("datetime").sort_index()
    window = first_contiguous_window(df, WINDOW_HOURS)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(window.index, window["actual_volume"],    color=ACTUAL_COLOR, label="Messwert")
    ax.plot(window.index, window["predicted_volume"], color=PRED_COLOR,   label="Vorhersage")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Fahrzeuge/h")
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.legend()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
