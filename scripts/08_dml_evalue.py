"""
08_dml_evalue.py
----------------
Double Machine Learning confounder-adjusted sensitivity analysis with E-value.
Reported in Additional file 1; summarised in one paragraph of the main text.

Usage:
    python scripts/08_dml_evalue.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os, math

SEED = 42
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFOUNDERS = ["GA", "age"]
OUTCOMES    = ["pH", "BE"]
UA_PI_THRESHOLD_PERCENTILE = 95   # 95th percentile of normal-Doppler group
N_BOOT = 500


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def dml_ate(df, outcome, treatment_col, confounders, seed=SEED):
    """
    Double Machine Learning estimate of the ATE of binary treatment on outcome.
    Stage 1: predict outcome and treatment from confounders (ridge regression).
    Stage 2: regress residuals on each other (OLS).
    """
    X_conf = df[confounders].values
    y      = df[outcome].values
    t      = df[treatment_col].values.astype(float)

    sc = StandardScaler()
    X_sc = sc.fit_transform(X_conf)

    # Stage 1 — outcome nuisance
    m_y = Ridge(alpha=1.0, random_state=seed)
    m_y.fit(X_sc, y)
    resid_y = y - m_y.predict(X_sc)

    # Stage 1 — treatment nuisance
    m_t = Ridge(alpha=1.0, random_state=seed)
    m_t.fit(X_sc, t)
    resid_t = t - m_t.predict(X_sc)

    # Stage 2 — OLS
    ate = np.dot(resid_t, resid_y) / np.dot(resid_t, resid_t)
    return ate, resid_y, resid_t


def bootstrap_ci(df, outcome, treatment_col, confounders,
                 n_boot=N_BOOT, seed=SEED):
    rng  = np.random.default_rng(seed)
    ates = []
    for _ in range(n_boot):
        idx  = rng.integers(0, len(df), size=len(df))
        boot = df.iloc[idx].copy()
        ate, _, _ = dml_ate(boot, outcome, treatment_col, confounders)
        ates.append(ate)
    return np.percentile(ates, [2.5, 97.5]), np.std(ates)


def evalue(ate, sd, ci_lower_bound=None):
    """
    E-value for linear outcome using Chinn approximation.
    OR ≈ exp(π * d / sqrt(3))  where d = ATE / SD
    E-value = OR + sqrt(OR * (OR - 1))
    """
    d = abs(ate) / sd
    OR = math.exp(math.pi * d / math.sqrt(3))
    ev = OR + math.sqrt(OR * (OR - 1))

    ev_ci = None
    if ci_lower_bound is not None:
        d_ci = abs(ci_lower_bound) / sd
        OR_ci = math.exp(math.pi * d_ci / math.sqrt(3))
        ev_ci = OR_ci + math.sqrt(OR_ci * (OR_ci - 1))

    return ev, ev_ci


def main():
    df = load_data()

    # Define binary treatment: UA/PI > 95th percentile of normal-Doppler group
    threshold = df.loc[df["group_doppler"] == 0, "UA_PI"].quantile(
        UA_PI_THRESHOLD_PERCENTILE / 100)
    df["high_uapi"] = (df["UA_PI"] > threshold).astype(int)
    print(f"Treatment threshold (UA/PI > {UA_PI_THRESHOLD_PERCENTILE}th "
          f"percentile of normal-Doppler): {threshold:.3f}")
    print(f"Treated (high_uapi=1): n={df.high_uapi.sum()}")

    sd_dict = {"pH": df["pH"].std(), "BE": df["BE"].std()}

    rows = []
    for out in OUTCOMES:
        ate, _, _ = dml_ate(df, out, "high_uapi", CONFOUNDERS)
        ci, se = bootstrap_ci(df, out, "high_uapi", CONFOUNDERS)
        ci_lower = min(abs(ci[0]), abs(ci[1]))  # conservative (smaller |bound|)

        ev_point, ev_ci = evalue(ate, sd_dict[out], ci_lower_bound=ci_lower)
        significant = not (ci[0] <= 0 <= ci[1])

        rows.append({
            "Outcome":        out,
            "ATE":            round(ate, 4),
            "CI_95_lower":    round(ci[0], 4),
            "CI_95_upper":    round(ci[1], 4),
            "SE_bootstrap":   round(se, 4),
            "Significant":    "Yes" if significant else "No",
            "Evalue_point":   round(ev_point, 2),
            "Evalue_CI_bound": round(ev_ci, 2) if ev_ci else None,
        })
        print(f"\n{out}: ATE={ate:.4f}, CI=({ci[0]:.4f}, {ci[1]:.4f}), "
              f"E-value (point)={ev_point:.2f}, E-value (CI bound)={ev_ci:.2f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS_DIR, "Tab5_DML_Evalue.csv"), index=False)
    print("\nTable 5 — DML + E-value (confounder-adjusted sensitivity analysis):")
    print(tab.to_string(index=False))
    print(f"\nNote: adjusts for GA and maternal age only. "
          f"Unmeasured confounders (hypertensive disorders, diabetes, "
          f"corticosteroids) were unavailable.")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
