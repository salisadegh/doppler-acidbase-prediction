"""
04_conformal_pi.py
------------------
Split conformal quantile regression for calibrated prediction intervals.
Produces Table 4 and Figure 5 (calibration curves).

Usage:
    python scripts/04_conformal_pi.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTCOMES = ["pH", "BE", "pCO2"]
PREDICTORS = ["GA", "age", "UA_PI"]
OUTCOME_CLASS = "acidemia_720"
OUTER_FOLDS = 5
CALIB_FRAC = 0.25
NOMINAL_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
ALPHA_QR = 0.1


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def fit_conformal_model(X_train, y_train, X_calib, y_calib, nominal=0.90):
    """
    Fit quantile regression (5th and 95th percentile) on training data.
    Compute conformal correction from calibration data.
    Returns (lower, upper, median) for X_calib and the correction margin.
    """
    lower_q = (1 - nominal) / 2
    upper_q = 1 - lower_q

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_cal_sc = sc.transform(X_calib)

    qr_low  = QuantileRegressor(quantile=lower_q, alpha=ALPHA_QR,
                                solver="highs").fit(X_tr_sc, y_train)
    qr_high = QuantileRegressor(quantile=upper_q,  alpha=ALPHA_QR,
                                solver="highs").fit(X_tr_sc, y_train)
    qr_med  = QuantileRegressor(quantile=0.50,     alpha=ALPHA_QR,
                                solver="highs").fit(X_tr_sc, y_train)

    lo_cal = qr_low.predict(X_cal_sc)
    hi_cal = qr_high.predict(X_cal_sc)
    med_cal = qr_med.predict(X_cal_sc)

    # Non-conformity scores
    scores = np.maximum(lo_cal - y_calib, y_calib - hi_cal)
    correction = np.quantile(scores, nominal)

    lo_adj = lo_cal - correction
    hi_adj = hi_cal + correction

    coverage = np.mean((y_calib >= lo_adj) & (y_calib <= hi_adj))
    mean_width = np.mean(hi_adj - lo_adj)

    return {
        "coverage": coverage,
        "mean_width": mean_width,
        "correction": correction,
        "lo": lo_adj, "hi": hi_adj, "med": med_cal,
        "models": (qr_low, qr_high, qr_med, sc),
    }


def evaluate_coverage_at_levels(X_train, y_train, X_calib, y_calib,
                                 levels=NOMINAL_LEVELS):
    """Evaluate achieved coverage at multiple nominal levels."""
    results = {}
    for nom in levels:
        res = fit_conformal_model(X_train, y_train, X_calib, y_calib, nom)
        results[nom] = res["coverage"]
    return results


def main():
    df = load_data()
    X = df[PREDICTORS].values
    y_dict = {out: df[out].values for out in OUTCOMES}
    strat = df[OUTCOME_CLASS].values

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                               random_state=SEED)

    coverage_results = {out: {nom: [] for nom in NOMINAL_LEVELS}
                        for out in OUTCOMES}
    width_90 = {out: [] for out in OUTCOMES}
    width_95 = {out: [] for out in OUTCOMES}

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, strat)):
        X_train, X_test = X[train_idx], X[test_idx]

        # Split training into train + calibration
        np.random.seed(SEED + fold_i)
        n_calib = max(10, int(len(train_idx) * CALIB_FRAC))
        calib_idx = np.random.choice(len(train_idx), n_calib, replace=False)
        fit_mask = np.ones(len(train_idx), dtype=bool)
        fit_mask[calib_idx] = False

        X_fit   = X_train[fit_mask]
        X_calib = X_train[calib_idx]

        for out in OUTCOMES:
            y_train = y_dict[out][train_idx]
            y_fit   = y_train[fit_mask]
            y_calib = y_train[calib_idx]

            cov_dict = evaluate_coverage_at_levels(
                X_fit, y_fit, X_calib, y_calib, NOMINAL_LEVELS)
            for nom, cov in cov_dict.items():
                coverage_results[out][nom].append(cov)

            res_90 = fit_conformal_model(X_fit, y_fit, X_calib, y_calib, 0.90)
            res_95 = fit_conformal_model(X_fit, y_fit, X_calib, y_calib, 0.95)
            width_90[out].append(res_90["mean_width"])
            width_95[out].append(res_95["mean_width"])

    # ── Table 4 ────────────────────────────────────────────────────────────
    table4_rows = []
    for out in OUTCOMES:
        cov_90 = np.mean(coverage_results[out][0.90])
        cov_95 = np.mean(coverage_results[out][0.95])
        w_90   = np.mean(width_90[out])
        w_95   = np.mean(width_95[out])
        table4_rows.append({
            "Outcome": out,
            "Coverage_90pct": round(cov_90, 3),
            "Coverage_95pct": round(cov_95, 3),
            "MeanWidth_90pct": round(w_90, 4),
            "MeanWidth_95pct": round(w_95, 4),
        })

    tab4 = pd.DataFrame(table4_rows)
    tab4.to_csv(os.path.join(RESULTS_DIR, "Tab4_PI_coverage.csv"), index=False)
    print("Table 4 — Prediction interval coverage:")
    print(tab4.to_string(index=False))

    # ── Figure 5: Calibration curves ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Fig. 5 — Calibration of Conformal Prediction Intervals\n"
        "(Cross-validated; N=222 development cohort)",
        fontsize=11, fontweight="bold",
    )

    for ax, out in zip(axes, OUTCOMES):
        nominal = np.array(NOMINAL_LEVELS)
        achieved = np.array([np.mean(coverage_results[out][n])
                             for n in NOMINAL_LEVELS])

        ax.plot([0.4, 1.0], [0.4, 1.0], "--", color="#888888",
                lw=1.5, label="Perfect calibration")
        ax.fill_between(nominal,
                        np.clip(nominal - 0.05, 0, 1),
                        np.clip(nominal + 0.05, 0, 1),
                        alpha=0.12, color="#1F6BB5", label="±5 pp band")
        ax.plot(nominal, achieved, "o-", color="#1F6BB5", lw=2.0,
                markersize=5, markerfacecolor="white", markeredgewidth=2,
                label="Conformal QR")

        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("Achieved Coverage" if out == "pH" else "")
        ax.set_title({"pH": "Umbilical Artery pH",
                      "BE": "Base Excess (mEq/L)",
                      "pCO2": "pCO₂ (mmHg)"}[out], fontweight="bold")
        ax.set_xlim(0.42, 1.01)
        ax.set_ylim(0.42, 1.01)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if out == "pH":
            ax.legend(loc="upper left", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig5_CalibrationCurves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
