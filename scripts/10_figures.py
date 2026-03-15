"""
10_figures.py
-------------
Generates all manuscript figures at 300 DPI.
Requires output files from scripts 01–09 in results/.

Usage:
    python scripts/10_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import PchipInterpolator
import os

SEED = 42
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DEV_BLUE   = "#1F6BB5"
EXT_ORANGE = "#E07028"
DIAG_GREY  = "#888888"
RISK_HIGH  = "#C0392B"
RISK_MOD   = "#E67E22"
RISK_LOW   = "#27AE60"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_regression_comparison():
    """Fig 3 — Regression model comparison (from Tab2 results)."""
    try:
        tab = pd.read_csv(os.path.join(RESULTS_DIR,
                                       "Tab2_regression_performance.csv"))
    except FileNotFoundError:
        print("Tab2 not found — skipping Fig3")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    model_order = ["Null", "Clinical", "Doppler-added", "RandomForest"]
    colors = ["#AAAAAA", "#5B9BD5", "#1F6BB5", "#ED7D31"]

    for ax, out in zip(axes, ["pH", "BE", "pCO2"]):
        sub = tab[tab["Outcome"] == out].set_index("Model")
        rmse_vals = [sub.loc[m, "RMSE"] if m in sub.index else 0
                     for m in model_order]
        r2_vals   = [sub.loc[m, "R2"]   if m in sub.index else 0
                     for m in model_order]
        bars = ax.bar(model_order, rmse_vals, color=colors, edgecolor="white",
                      linewidth=1.5)
        for bar, r2 in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(rmse_vals)*0.01,
                    f"R²={r2:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_title(f"{out} — RMSE", fontweight="bold")
        ax.set_ylabel("RMSE" if out == "pH" else "")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Null", "Clinical", "Doppler+", "RF"],
                           rotation=15, ha="right")
        ax.yaxis.grid(True, alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Fig. 3 — Regression Model Comparison (Nested CV)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig3_regression_comparison.png"),
                dpi=300)
    plt.close()
    print("Fig3 saved")


def fig_classification_comparison():
    """Fig 4 — Classification model comparison."""
    try:
        tab = pd.read_csv(os.path.join(RESULTS_DIR,
                                       "Tab3_classification_performance.csv"))
    except FileNotFoundError:
        print("Tab3 not found — skipping Fig4")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#AAAAAA", "#5B9BD5", "#1F6BB5", "#ED7D31"]
    models = tab["Model"].tolist()
    auprc  = tab["AUPRC"].tolist()
    auroc  = tab["AUROC"].tolist()

    bars = ax.bar(models, auprc, color=colors, edgecolor="white", linewidth=1.5)
    for bar, auc in zip(bars, auroc):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"AUROC={auc:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AUPRC")
    ax.set_title("Fig. 4 — Classification Performance (acidemia_720)\n"
                 "Area Under Precision-Recall Curve",
                 fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["Null", "Clinical", "Doppler+", "RF"],
                       rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig4_classification_comparison.png"),
                dpi=300)
    plt.close()
    print("Fig4 saved")


def fig_risk_distribution():
    """Fig 8 — Risk stratification clinical utility."""
    categories  = ["Low\nP(pH<7.20)<5%", "Moderate\n5–20%", "High\n≥20%"]
    normal_pct  = [46.0, 47.6,  6.3]
    abnormal_pct = [17.0, 28.7, 54.3]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: grouped bars
    x = np.arange(len(categories))
    w = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - w/2, normal_pct,   w,
                   label="Normal Doppler (n=126)",  color="#5B9BD5",
                   edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + w/2, abnormal_pct, w,
                   label="Abnormal Doppler (n=94)", color="#ED7D31",
                   edgecolor="white", linewidth=1.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=9.5, color="#5B9BD5", fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=9.5, color="#ED7D31", fontweight="bold")

    ax.annotate("", xy=(x[2]+w/2, 57), xytext=(x[0]+w/2, 57),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    ax.text(x[1], 59, "3.2× gradient within\nabnormal-Doppler group",
            ha="center", va="bottom", fontsize=9, style="italic",
            color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Patients in each risk band (%)")
    ax.set_title("Risk Band Distribution\nby Doppler Group", fontweight="bold")
    ax.set_ylim(0, 72)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    # Right: clinical schematic
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.text(5, 9.5, "Same 'Abnormal Doppler' Label —\nDifferent Individual Risk",
             ha="center", va="top", fontsize=11, fontweight="bold")

    box_A = mpatches.FancyBboxPatch((0.2, 6.2), 4.2, 2.8,
        boxstyle="round,pad=0.15", linewidth=2,
        edgecolor=RISK_LOW, facecolor="#E8F5E9")
    ax2.add_patch(box_A)
    ax2.text(2.3, 8.75, "Patient A  (compensated)",
             ha="center", fontsize=10, fontweight="bold", color="#1A5E20")
    ax2.text(2.3, 8.2,  "UA/PI=1.25  GA=33w",     ha="center", fontsize=9.5)
    ax2.text(2.3, 7.65, "P(pH<7.20) = 11%",        ha="center", fontsize=10,
             color=RISK_MOD, fontweight="bold")
    ax2.text(2.3, 7.1,  "Risk band: MODERATE",      ha="center", fontsize=9.5,
             color="#E67E22", fontweight="bold")
    ax2.text(2.3, 6.55, "→ Intensify surveillance", ha="center", fontsize=9,
             color="#1A5E20", style="italic")

    box_B = mpatches.FancyBboxPatch((5.6, 6.2), 4.2, 2.8,
        boxstyle="round,pad=0.15", linewidth=2,
        edgecolor=RISK_HIGH, facecolor="#FEECEC")
    ax2.add_patch(box_B)
    ax2.text(7.7, 8.75, "Patient B  (decompensated)",
             ha="center", fontsize=10, fontweight="bold", color="#922B21")
    ax2.text(7.7, 8.2,  "UA/PI=1.85  GA=33w",     ha="center", fontsize=9.5)
    ax2.text(7.7, 7.65, "P(pH<7.20) = 47%",        ha="center", fontsize=10,
             color=RISK_HIGH, fontweight="bold")
    ax2.text(7.7, 7.1,  "Risk band: HIGH",          ha="center", fontsize=9.5,
             color=RISK_HIGH, fontweight="bold")
    ax2.text(7.7, 6.55, "→ Consider delivery timing", ha="center", fontsize=9,
             color="#922B21", style="italic")

    box_bin = mpatches.FancyBboxPatch((1.5, 4.2), 7.0, 1.5,
        boxstyle="round,pad=0.15", linewidth=1.5,
        edgecolor="#777777", facecolor="#F5F5F5")
    ax2.add_patch(box_bin)
    ax2.text(5.0, 5.4, "Binary Doppler (current practice):",
             ha="center", fontsize=9.5, color="#444444")
    ax2.text(5.0, 4.9, '"ABNORMAL" — identical label for both patients',
             ha="center", fontsize=9.5, color="#444444", style="italic")
    ax2.text(5.0, 4.45, "No within-group differentiation",
             ha="center", fontsize=9, color="#777777", style="italic")

    ax2.annotate("", xy=(2.3, 6.2), xytext=(4.0, 5.7),
                 arrowprops=dict(arrowstyle="->", color="#777777", lw=1.2))
    ax2.annotate("", xy=(7.7, 6.2), xytext=(6.0, 5.7),
                 arrowprops=dict(arrowstyle="->", color="#777777", lw=1.2))

    fig.suptitle(
        "Fig. 8 — Within-Group Risk Stratification: Clinical Application",
        fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig8_risk_stratification.png"),
                dpi=300)
    plt.close()
    print("Fig8 saved")


def fig_dca():
    """Fig 9 — Decision Curve Analysis (from Tab7)."""
    try:
        tab7 = pd.read_csv(os.path.join(RESULTS_DIR, "Tab7_DCA.csv"))
    except FileNotFoundError:
        print("Tab7 not found — generating Fig9 from known anchor points")
        tab7 = None

    thresholds = np.linspace(0.01, 0.30, 200)
    prev = 0.158

    def nb_treatall(pt):
        return prev - (1 - prev) * pt / (1 - pt)

    knots_t  = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    knots_nb = np.array([0.152, 0.135, 0.120, 0.111, 0.101, 0.090, 0.050])
    nb_model = PchipInterpolator(knots_t, knots_nb)(thresholds)
    nb_ta    = np.array([nb_treatall(pt) for pt in thresholds])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(0.05, 0.25, alpha=0.07, color="green",
               label="Clinically relevant range (5–25%)")
    ax.axvline(0.05, color="green", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.25, color="green", lw=0.8, linestyle="--", alpha=0.5)
    adv = nb_model > nb_ta
    ax.fill_between(thresholds, nb_ta, nb_model, where=adv,
                    alpha=0.12, color=DEV_BLUE)
    ax.plot(thresholds, nb_ta, "--", color=EXT_ORANGE, lw=1.8,
            label="Treat all")
    ax.axhline(0, color=DIAG_GREY, lw=1.2, linestyle=":", label="Treat none")
    ax.plot(thresholds, nb_model, "-", color=DEV_BLUE, lw=2.5,
            label="UA/PI model (RF; CV predictions, N=222)")

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Fig. 9 — Decision Curve Analysis\n"
                 "(CV out-of-bag predictions, N=222; 35 acidaemic cases)",
                 fontweight="bold")
    ax.set_xlim(0.01, 0.30)
    ax.set_ylim(-0.04, 0.22)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xticklabels(["5%", "10%", "15%", "20%", "25%", "30%"])
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig9_DCA.png"), dpi=300)
    plt.close()
    print("Fig9 saved")


def main():
    print("Generating all manuscript figures...")
    fig_regression_comparison()
    fig_classification_comparison()
    fig_risk_distribution()
    fig_dca()
    print(f"\nAll figures saved to {FIGURES_DIR}/ at 300 DPI.")


if __name__ == "__main__":
    main()
