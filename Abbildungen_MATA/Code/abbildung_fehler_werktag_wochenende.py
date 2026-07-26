"""
Abb. 24: Relativer Vorhersagefehler des DNN (v8) nach Werktag und Wochenende
je Zaehlstelle. MAE geteilt durch das mittlere Volumen der jeweiligen Teilmenge,
damit die Stationen mit unterschiedlichem Verkehrsniveau vergleichbar sind.
Quelle der Daten: results/model_results/mlp_v8/predictions/ (Testset).
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED = "results/model_results/mlp_v8/predictions"
STATIONS = {"299_Wollerau": "Wollerau", "171_Sattel": "Sattel",
            "720_Schwyz": "Schwyz", "216_Wangen": "Wangen SZ", "050_Brunnen": "Brunnen"}
COLOR_WD, COLOR_WK = "#3182bd", "#9ecae1"


def rel_error(sub):
    return 100 * sub["err"].mean() / sub["actual_volume"].mean()


def main():
    data = {}
    for key, name in STATIONS.items():
        df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{PRED}/predictions_{key}*_v8.csv")],
                       ignore_index=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["err"] = (df["actual_volume"] - df["predicted_volume"]).abs()
        we = df["datetime"].dt.dayofweek >= 5
        data[name] = (rel_error(df[~we]), rel_error(df[we]))

    order = sorted(data, key=lambda n: data[n][0])
    wd = [data[n][0] for n in order]
    wk = [data[n][1] for n in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, wd, w, label="Werktag", color=COLOR_WD)
    b2 = ax.bar(x + w / 2, wk, w, label="Wochenende", color=COLOR_WK)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.25,
                    f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Relativer Fehler (MAE / mittleres Volumen) in %")
    ax.set_ylim(0, max(wk) + 3)
    ax.set_title("Relativer Vorhersagefehler des DNN nach Werktag und Wochenende je Zaehlstelle",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("Abbildungen_MATA/Abbildungen/abbildung_fehler_werktag_wochenende_v8.png",
                dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
