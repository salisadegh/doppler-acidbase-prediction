"""
07_dca.py
---------
Cross-validated Decision Curve Analysis on N=222 (out-of-bag RF predictions).
Produces Table 7 and Figure 9.

Usage:
    python scripts/07_dca.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

SEED = 42
DATA_DIR    = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PREDICTORS = ["GA", "age", "UA_PI"]
OUTCOME_CLASS = "acidemia_720"
OUTER_FOLDS = 5
THRESHOLDS = np.linspace(0.01, 0.30, 200)
CLINICAL_RANGE = (0.05, 0.25)
REPORT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def net_benefit(y_true, y_prob, pt):
    """Net benefit at threshold probability pt."""
    N = len(y_true)
    pred_pos = y_prob >= pt
    TP = np.sum(pred_pos & (y_true == 1))
    FP = np.sum(pred_pos & (y_true == 0))
    return TP / N - FP / N * pt / (1 - pt)


def net_benefit_treatall(y_true, pt):
    N = len(y_true)
    prev = y_true.mean()
    return prev - (1 - prev) * pt / (1 - pt)


def main():
    df = load_data()
    X  = df[PREDICTORS].values
    y  = df[OUTCOME_CLASS].values

    # Cross-validated OOB predictions
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                         random_state=SEED)
    model = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=SEED, n_jobs=-1)),
    ])
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    # ── NB curves ──────────────────────────────────────────────────────────
    nb_model    = np.array([net_benefit(y, probs, pt) for pt in THRESHOLDS])
    nb_treatall = np.array([net_benefit_treatall(y, pt) for pt in THRESHOLDS])
    nb_treatnone = np.zeros_like(THRESHOLDS)

    # ── Table 7 ────────────────────────────────────────────────────────────
    rows = []
    for pt in REPORT_THRESHOLDS:
        idx = np.argmin(np.abs(THRESHOLDS - pt))
        rows.append({
            "Threshold": f"{int(pt*100)}%",
            "NB_Model": round(nb_model[idx], 4),
            "NB_TreatAll": round(nb_treatall[idx], 4),
            "NB_TreatNone": 0.000,
            "Advantage": round(nb_model[idx] - nb_treatall[idx], 4),
        })

    tab7 = pd.DataFrame(rows)
    tab7.to_csv(os.path.join(RESULTS_DIR, "Tab7_DCA.csv"), index=False)
    print("Table 7 — Cross-validated DCA:")
    print(tab7.to_string(index=False))

    # ── Figure 9 ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Clinical range shading
    ax.axvspan(CLINICAL_RANGE[0], CLINICAL_RANGE[1],
               alpha=0.07, color="green", label="Clinically relevant range (5–25%)")
    ax.axvline(CLINICAL_RANGE[0], color="green", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(CLINICAL_RANGE[1], color="green", lw=0.8, linestyle="--", alpha=0.5)

    # Model advantage zone
    adv = nb_model > nb_treatall
    ax.fill_between(THRESHOLDS, nb_treatall, nb_model,
                    where=adv, alpha=0.12, color="#1F6BB5")

    ax.plot(THRESHOLDS, nb_treatall, "--", color="#E07028", lw=1.8,
            label="Treat all")
    ax.axhline(0, color="#888888", lw=1.2, linestyle=":", label="Treat none")
    ax.plot(THRESHOLDS, nb_model, "-", color="#1F6BB5", lw=2.5,
            label="UA/PI model (RF; CV predictions, N=222)")

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(
        "Fig. 9 — Decision Curve Analysis\n"
        "(Cross-validated out-of-bag predictions; N=222; 35 acidaemic cases)",
        fontweight="bold",
    )
    ax.set_xlim(0.01, 0.30)
    ax.set_ylim(-0.04, 0.22)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xticklabels(["5%", "10%", "15%", "20%", "25%", "30%"])
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig_path = os.path.join(FIGURES_DIR, "Fig9_DCA.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
