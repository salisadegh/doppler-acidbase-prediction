"""
02_eda.py
---------
Exploratory data analysis: baseline characteristics table and diagnostic plots.
Produces Table 1 (baseline), Figure 1 (distributions), Figure 2 (scatter).

Usage:
    python scripts/02_eda.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

SEED = 42
DATA_DIR    = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CONTINUOUS_VARS = ["pH", "pCO2", "BE", "UA_PI", "GA", "age"]
OUTCOME_CLASS   = "acidemia_720"


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def cohens_d(a, b):
    pooled_sd = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0


def describe_group(series):
    return {
        "N":      series.notna().sum(),
        "Mean":   round(series.mean(), 3),
        "SD":     round(series.std(ddof=1), 3),
        "Median": round(series.median(), 3),
        "Q1":     round(series.quantile(0.25), 3),
        "Q3":     round(series.quantile(0.75), 3),
    }


def compare_groups(df_n, df_a, var):
    """Mann-Whitney or t-test depending on normality."""
    a = df_n[var].dropna()
    b = df_a[var].dropna()

    _, p_n = stats.shapiro(a[:50])  # Shapiro on first 50 for speed
    _, p_a = stats.shapiro(b[:50])

    if p_n > 0.05 and p_a > 0.05 and var == "age":
        stat, p = stats.ttest_ind(a, b)
        test = "t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test = "Mann-Whitney"

    d = abs(cohens_d(a, b))
    return test, round(stat, 3), round(p, 6), round(d, 3)


def make_table1(df):
    df_n = df[df["group_doppler"] == 0]
    df_a = df[df["group_doppler"] == 1]

    rows = []
    for var in CONTINUOUS_VARS:
        desc_n = describe_group(df_n[var])
        desc_a = describe_group(df_a[var])
        desc_all = describe_group(df[var])
        test, stat, pval, d = compare_groups(df_n, df_a, var)
        rows.append({
            "Variable": var,
            "Normal_N": desc_n["N"],
            "Normal_Mean": desc_n["Mean"], "Normal_SD": desc_n["SD"],
            "Normal_Median": desc_n["Median"],
            "Abnormal_N": desc_a["N"],
            "Abnormal_Mean": desc_a["Mean"], "Abnormal_SD": desc_a["SD"],
            "Abnormal_Median": desc_a["Median"],
            "All_N": desc_all["N"],
            "Test": test, "Statistic": stat,
            "p_value": pval, "Cohen_d": d,
        })

    # Binary: acidemia
    prev_n   = df_n[OUTCOME_CLASS].mean()
    prev_a   = df_a[OUTCOME_CLASS].mean()
    prev_all = df[OUTCOME_CLASS].mean()
    rows.append({
        "Variable": "acidemia_720 (%)",
        "Normal_N": len(df_n), "Normal_Mean": round(prev_n*100, 1),
        "Abnormal_N": len(df_a), "Abnormal_Mean": round(prev_a*100, 1),
        "All_N": len(df), "All_Mean": round(prev_all*100, 1),
    })

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS_DIR, "Tab1_baseline_characteristics.csv"),
               index=False)
    return tab


def make_figure1(df):
    """Violin plots: normal vs abnormal Doppler for each variable."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    labels = {
        "pH": "Umbilical Artery pH",
        "pCO2": "pCO₂ (mmHg)",
        "BE": "Base Excess (mEq/L)",
        "UA_PI": "UA/PI",
        "GA": "Gestational Age (weeks)",
        "age": "Maternal Age (years)",
    }

    for ax, var in zip(axes, CONTINUOUS_VARS):
        data_n = df.loc[df["group_doppler"] == 0, var].dropna()
        data_a = df.loc[df["group_doppler"] == 1, var].dropna()

        parts = ax.violinplot([data_n, data_a], positions=[0, 1],
                              showmedians=True, showextrema=False)
        parts["bodies"][0].set_facecolor("#5B9BD5")
        parts["bodies"][0].set_alpha(0.7)
        parts["bodies"][1].set_facecolor("#ED7D31")
        parts["bodies"][1].set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)

        _, p = stats.mannwhitneyu(data_n, data_a, alternative="two-sided")
        p_str = f"p<0.001" if p < 0.001 else f"p={p:.4f}"
        ax.text(0.97, 0.97, p_str, ha="right", va="top",
                transform=ax.transAxes, fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal Doppler", "Abnormal Doppler"])
        ax.set_title(labels.get(var, var), fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 1 — Distribution of Variables by Doppler Group",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig1_group_distributions.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def make_figure2(df):
    """Scatter: UA/PI vs each acid-base outcome, coloured by Doppler group."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    y_labels = {"pH": "Umbilical Artery pH",
                "BE": "Base Excess (mEq/L)",
                "pCO2": "pCO₂ (mmHg)"}
    colors = {0: "#5B9BD5", 1: "#ED7D31"}
    group_labels = {0: "Normal Doppler", 1: "Abnormal Doppler"}

    for ax, out in zip(axes, ["pH", "BE", "pCO2"]):
        for grp in [0, 1]:
            sub = df[df["group_doppler"] == grp]
            ax.scatter(sub["UA_PI"], sub[out],
                       c=colors[grp], alpha=0.6, s=25,
                       label=group_labels[grp], edgecolors="none")
        ax.set_xlabel("UA/PI")
        ax.set_ylabel(y_labels[out])
        ax.set_title(f"UA/PI vs {out}", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if out == "pCO2":
            ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Fig. 2 — Scatter Plots: UA/PI vs Acid–Base Outcomes",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig2_scatter_uapi_vs_abg.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = load_data()
    print(f"N={len(df)}: normal={( df.group_doppler==0).sum()}, "
          f"abnormal={(df.group_doppler==1).sum()}")

    tab1 = make_table1(df)
    print("\nTable 1 — Baseline characteristics:")
    print(tab1[["Variable", "Normal_Mean", "Normal_SD",
                "Abnormal_Mean", "Abnormal_SD", "p_value", "Cohen_d"
                ]].to_string(index=False))

    make_figure1(df)
    make_figure2(df)
    print(f"\nFigures saved to {FIGURES_DIR}/")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
