"""
Abb. 25: Relativer Vorhersagefehler des DNN (v8) nach Normal- und Ferientag
je Zaehlstelle. Ferientag = Schulferien oder Feiertag im Kanton Schwyz
(Flags schoolholiday_SZ / holiday_SZ aus den v8-Daten, ueber den Zeitstempel
mit den Vorhersagen verbunden). MAE geteilt durch das mittlere Volumen der
jeweiligen Teilmenge. Datenquelle: results/model_results/mlp_v8/predictions/
und data/v8/ (Testset).
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED = "results/model_results/mlp_v8/predictions"
DATA = "data/v8"
STATIONS = {"299_Wollerau": "Wollerau", "171_Sattel": "Sattel",
            "720_Schwyz": "Schwyz", "216_Wangen": "Wangen SZ", "050_Brunnen": "Brunnen"}
COLOR_N, COLOR_F = "#3182bd", "#9ecae1"


def rel_error(sub):
    return 100 * sub["err"].mean() / sub["actual_volume"].mean()


def main():
    data = {}
    for key, name in STATIONS.items():
        dfs = []
        for pf in sorted(glob.glob(f"{PRED}/predictions_{key}*_v8.csv")):
            base = pf.split("predictions_")[1]
            dd = pd.read_csv(f"{DATA}/{base}", usecols=["datetime", "holiday_SZ", "schoolholiday_SZ"])
            dd["datetime"] = pd.to_datetime(dd["datetime"])
            p = pd.read_csv(pf); p["datetime"] = pd.to_datetime(p["datetime"])
            dfs.append(p.merge(dd, on="datetime", how="left"))
        df = pd.concat(dfs, ignore_index=True)
        df["err"] = (df["actual_volume"] - df["predicted_volume"]).abs()
        df["ferien"] = (df["holiday_SZ"] == 1) | (df["schoolholiday_SZ"] == 1)
        data[name] = (rel_error(df[~df["ferien"]]), rel_error(df[df["ferien"]]))

    order = sorted(data, key=lambda n: data[n][0])
    nt = [data[n][0] for n in order]
    ft = [data[n][1] for n in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, nt, w, label="Normaltag", color=COLOR_N)
    b2 = ax.bar(x + w / 2, ft, w, label="Ferientag (Schulferien oder Feiertag, Kt. SZ)", color=COLOR_F)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.3,
                    f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Relativer Fehler (MAE / mittleres Volumen) in %")
    ax.set_ylim(0, max(ft) + 3)
    ax.set_title("Relativer Vorhersagefehler des DNN nach Normal- und Ferientag je Zaehlstelle",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("Abbildungen_MATA/Abbildungen/abbildung_fehler_normaltag_ferientag_v8.png",
                dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
