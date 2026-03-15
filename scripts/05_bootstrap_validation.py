"""
05_bootstrap_validation.py
--------------------------
Bootstrap internal validation with optimism correction (1000 resamples).
Produces Table 8 from the manuscript.

Usage:
    python scripts/05_bootstrap_validation.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, roc_auc_score, average_precision_score, brier_score_loss,
)
import os

SEED   = 42
N_BOOT = 1000
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTCOMES_REG  = ["pH", "BE", "pCO2"]
OUTCOME_CLASS = "acidemia_720"
PREDICTORS    = ["GA", "age", "UA_PI"]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def make_ridge():
    return Pipeline([("sc", StandardScaler()),
                     ("r", Ridge(alpha=1.0, random_state=SEED))])

def make_rf_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(
                         n_estimators=200, class_weight="balanced",
                         random_state=SEED, n_jobs=-1))])

def make_lr():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(
                         C=1.0, class_weight="balanced",
                         solver="lbfgs", max_iter=1000, random_state=SEED))])


def bootstrap_regression(X, y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    apparent = []
    oob      = []
    for _ in range(n_boot):
        idx_boot = rng.integers(0, len(X), size=len(X))
        mask_oob = np.ones(len(X), dtype=bool)
        mask_oob[idx_boot] = False
        if mask_oob.sum() < 5:
            continue
        m = make_ridge()
        m.fit(X[idx_boot], y[idx_boot])
        r2_app = r2_score(y[idx_boot], m.predict(X[idx_boot]))
        r2_oob = r2_score(y[mask_oob], m.predict(X[mask_oob]))
        apparent.append(r2_app)
        oob.append(r2_oob)
    optimism = np.mean(apparent) - np.mean(oob)
    return np.mean(oob), optimism, np.percentile(oob, [2.5, 97.5])


def bootstrap_classifier(X, y, model_fn, metric_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    apparent = []
    oob      = []
    for _ in range(n_boot):
        idx_boot = rng.integers(0, len(X), size=len(X))
        mask_oob = np.ones(len(X), dtype=bool)
        mask_oob[idx_boot] = False
        if mask_oob.sum() < 5 or y[mask_oob].sum() < 2:
            continue
        m = model_fn()
        m.fit(X[idx_boot], y[idx_boot])
        prob_app = m.predict_proba(X[idx_boot])[:, 1]
        prob_oob = m.predict_proba(X[mask_oob])[:, 1]
        apparent.append(metric_fn(y[idx_boot], prob_app))
        oob.append(metric_fn(y[mask_oob], prob_oob))
    optimism = np.mean(apparent) - np.mean(oob)
    return np.mean(oob), optimism, np.percentile(oob, [2.5, 97.5])


def main():
    df = load_data()
    X  = df[PREDICTORS].values
    y_class = df[OUTCOME_CLASS].values

    rows = []

    # Regression outcomes
    cv_r2 = {"pH": 0.46, "BE": 0.72, "pCO2": 0.11}  # from script 03
    for out in OUTCOMES_REG:
        y = df[out].values
        boot_mean, optimism, ci = bootstrap_regression(X, y)
        opt_corrected = cv_r2[out] - optimism
        rows.append({
            "Metric": f"{out} R² — Doppler-added",
            "CV_estimate": cv_r2[out],
            "Bootstrap_mean": round(boot_mean, 3),
            "Bootstrap_95CI": f"{ci[0]:.2f}–{ci[1]:.2f}",
            "Optimism": round(optimism, 3),
            "Optimism_corrected": round(opt_corrected, 3),
        })
        print(f"{out} R²: CV={cv_r2[out]}, boot_mean={boot_mean:.3f}, "
              f"optimism={optimism:.3f}, corrected={opt_corrected:.3f}")

    # Classification — AUPRC (RF)
    cv_auprc = 0.855
    boot_mean_a, opt_a, ci_a = bootstrap_classifier(
        X, y_class, make_rf_clf, average_precision_score)
    rows.append({
        "Metric": f"Acidemia AUPRC — RF",
        "CV_estimate": cv_auprc,
        "Bootstrap_mean": round(boot_mean_a, 3),
        "Bootstrap_95CI": f"{ci_a[0]:.2f}–{ci_a[1]:.2f}",
        "Optimism": round(opt_a, 3),
        "Optimism_corrected": round(cv_auprc - opt_a, 3),
    })

    # Classification — AUROC (RF)
    cv_auroc = 0.913
    boot_mean_r, opt_r, ci_r = bootstrap_classifier(
        X, y_class, make_rf_clf, roc_auc_score)
    rows.append({
        "Metric": "Acidemia AUROC — RF",
        "CV_estimate": cv_auroc,
        "Bootstrap_mean": round(boot_mean_r, 3),
        "Bootstrap_95CI": f"{ci_r[0]:.2f}–{ci_r[1]:.2f}",
        "Optimism": round(opt_r, 3),
        "Optimism_corrected": round(cv_auroc - opt_r, 3),
    })

    # Brier (RF)
    cv_brier = 0.059
    boot_mean_b, opt_b, ci_b = bootstrap_classifier(
        X, y_class, make_rf_clf,
        lambda yt, yp: brier_score_loss(yt, yp))
    rows.append({
        "Metric": "Brier score — RF",
        "CV_estimate": cv_brier,
        "Bootstrap_mean": round(boot_mean_b, 3),
        "Bootstrap_95CI": f"{ci_b[0]:.3f}–{ci_b[1]:.3f}",
        "Optimism": round(opt_b, 3),
        "Optimism_corrected": round(cv_brier + opt_b, 3),  # Brier: lower is better
    })

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS_DIR, "Tab8_bootstrap_validation.csv"),
               index=False)
    print("\nTable 8 — Bootstrap internal validation:")
    print(tab.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
