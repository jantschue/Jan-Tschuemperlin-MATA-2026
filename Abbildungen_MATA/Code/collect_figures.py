"""
Dieses Hilfsskript sammelt alle im Projekt erzeugten Abbildungen der Maturaarbeit
(eigene Darstellungen) an einem einzigen Ort, damit sie übersichtlich beisammen
liegen. Die Grafiken werden aus ihren jeweiligen Ausgabeordnern (Abbildungen_MATA,
results/…) in den Zielordner Abbildungen_MATA/Alle_Abbildungen/ kopiert und dabei
nach ihrer Abbildungsnummer im Dokument benannt (z.B. Abb_09_...).

Voraussetzung: Die Abbildungen wurden zuvor erzeugt (die erzeugenden Skripte bzw.
die Pipeline sind gelaufen). Fehlende Quellen werden übersprungen und gemeldet.
Es werden nur relative Pfade verwendet.
"""

import os
import shutil

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Alle_Abbildungen")

ABB = os.path.join("Abbildungen_MATA", "Abbildungen")
LR8 = os.path.join("results", "model_results", "linear_regression_v8", "summary")
MLP8 = os.path.join("results", "model_results", "mlp_v8", "summary")
MLP8_TRAIN = os.path.join(
    "results", "model_results", "mlp_v8", "plots", "trainingsverlauf"
)
HOURLY = os.path.join("results", "data_visualizations", "hourly_plots")
CORR = os.path.join("results", "data_visualizations", "correlation_analysis")

# Zuordnung: Abbildungsnummer -> (Quelldatei, sprechender Zielname).
# Nur die aus Projekt-Skripten erzeugten Abbildungen (eigene Darstellungen);
# die externen Abbildungen 4, 5 und 6 stammen nicht aus dem Repository.
FIGURES = [
    (1,  os.path.join(ABB, "klassische_vs_ml_programmierung.png"), "klassische_vs_ml_programmierung"),
    (2,  os.path.join(ABB, "abbildung_2_regression_diagramm.png"), "suche_regressionsgerade"),
    (3,  os.path.join(ABB, "temperatur_glace_regression.png"), "lineare_regression_glace"),
    (7,  os.path.join(ABB, "abbildung_5_linear_vs_relu.png"), "linear_vs_relu"),
    (8,  os.path.join(ABB, "mlp_schema.png"), "mlp_schema"),
    (9,  os.path.join(HOURLY, "720_Schwyz_R1_v8_hourly_trend_clean.png"), "tagesverlauf_wetter_verkehr_SchwyzR1"),
    (10, os.path.join(ABB, "zaehlstellen_karte.png"), "zaehlstellen_karte"),
    (11, os.path.join(CORR, "720_Schwyz_R1_v8_correlation_clean.png"), "korrelationsmatrix_SchwyzR1"),
    (12, os.path.join(ABB, "abbildung_parameter_vs_performance_v8.png"), "netzgroesse_vs_test_r2_BrunnenR1"),
    (13, os.path.join(LR8, "r2_summary_clean.png"), "r2_lineare_regression"),
    (14, os.path.join(LR8, "timeseries_clean_Schwyz_R1.png"), "prognose_lineare_regression_SchwyzR1"),
    (15, os.path.join(MLP8, "r2_summary_clean.png"), "r2_dnn"),
    (16, os.path.join(MLP8, "timeseries_clean_Schwyz_R1.png"), "prognose_dnn_SchwyzR1"),
    (17, os.path.join(MLP8_TRAIN, "trainingsverlauf_r2_720_Schwyz_R1_v8.png"), "trainingsverlauf_dnn_SchwyzR1"),
    (18, os.path.join(ABB, "abbildung_prognose_vs_messwert_r1_v8.png"), "prognose_vs_messwert_raster_R1"),
    (19, os.path.join(ABB, "abbildung_r2_vergleich_lr_dnn_v8.png"), "r2_vergleich_lr_dnn"),
    (20, os.path.join(ABB, "abbildung_permutation_importance_v8.png"), "permutation_importance"),
    (21, os.path.join(ABB, "abbildung_r2_veraenderung_corona_v8.png"), "r2_veraenderung_corona"),
    (22, os.path.join(ABB, "abbildung_trainingszeit_lr_dnn_v8.png"), "trainingszeit_lr_dnn"),
    (23, os.path.join(ABB, "abbildung_genauigkeit_vs_aufwand_v8.png"), "genauigkeit_vs_aufwand"),
]


def collect_figures():
    """Kopiert alle vorhandenen Abbildungen nummeriert in den Sammelordner."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied, missing = 0, []
    for number, source, name in FIGURES:
        if not os.path.exists(source):
            missing.append((number, source))
            continue
        target = os.path.join(OUTPUT_DIR, f"Abb_{number:02d}_{name}.png")
        shutil.copy2(source, target)
        copied += 1
        print(f"Abb {number:>2}: {source} -> {target}")

    print(f"\n{copied} Abbildungen nach '{OUTPUT_DIR}' kopiert.")
    if missing:
        print(f"{len(missing)} Quelle(n) fehlten (Skripte/Pipeline zuerst ausfuehren):")
        for number, source in missing:
            print(f"  Abb {number}: {source}")


if __name__ == "__main__":
    collect_figures()
