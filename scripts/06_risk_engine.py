"""
06_risk_engine.py
-----------------
Monte Carlo individual risk scores from conformal prediction intervals.
Produces Table 6 (risk stratification) and per-patient risk score CSV.

Usage:
    python scripts/06_risk_engine.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import os

SEED = 42
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTORS = ["GA", "age", "UA_PI"]
OUTCOME_CLASS = "acidemia_720"
N_MC = 10_000
ALPHA_QR = 0.10
NOMINAL = 0.90
RISK_LOW_THR  = 0.05
RISK_HIGH_THR = 0.20


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def fit_pi_model(X_train, y_train, X_pred, nominal=NOMINAL):
    """Fit split-conformal quantile regression on training data."""
    n_calib = max(10, int(len(X_train) * 0.25))
    np.random.seed(SEED)
    calib_mask = np.zeros(len(X_train), dtype=bool)
    calib_mask[np.random.choice(len(X_train), n_calib, replace=False)] = True
    fit_mask = ~calib_mask

    sc = StandardScaler()
    X_fit_sc   = sc.fit_transform(X_train[fit_mask])
    X_calib_sc = sc.transform(X_train[calib_mask])
    X_pred_sc  = sc.transform(X_pred)

    lower_q = (1 - nominal) / 2
    upper_q = 1 - lower_q

    qr_lo  = QuantileRegressor(quantile=lower_q,  alpha=ALPHA_QR,
                               solver="highs").fit(X_fit_sc, y_train[fit_mask])
    qr_hi  = QuantileRegressor(quantile=upper_q,  alpha=ALPHA_QR,
                               solver="highs").fit(X_fit_sc, y_train[fit_mask])
    qr_med = QuantileRegressor(quantile=0.50,     alpha=ALPHA_QR,
                               solver="highs").fit(X_fit_sc, y_train[fit_mask])

    lo_cal = qr_lo.predict(X_calib_sc)
    hi_cal = qr_hi.predict(X_calib_sc)
    scores = np.maximum(lo_cal - y_train[calib_mask],
                        y_train[calib_mask] - hi_cal)
    correction = np.quantile(scores, nominal)

    med = qr_med.predict(X_pred_sc)
    lo  = qr_lo.predict(X_pred_sc) - correction
    hi  = qr_hi.predict(X_pred_sc) + correction
    return med, lo, hi


def monte_carlo_risk(med, lo, hi, threshold=7.20, n_mc=N_MC):
    """
    P(pH < threshold) via Monte Carlo sampling from Gaussian PI.
    σ = (upper - lower) / (2 * 1.645)
    """
    sigma = (hi - lo) / (2 * 1.645)
    sigma = np.maximum(sigma, 1e-6)  # avoid zero sigma
    rng = np.random.default_rng(SEED)
    samples = rng.normal(loc=med[:, None],
                         scale=sigma[:, None],
                         size=(len(med), n_mc))
    probs = (samples < threshold).mean(axis=1)
    return probs


def assign_risk_band(probs):
    bands = np.where(probs < RISK_LOW_THR, "Low",
             np.where(probs >= RISK_HIGH_THR, "High", "Moderate"))
    return bands


def main():
    df = load_data()
    X  = df[PREDICTORS].values
    y_pH = df["pH"].values
    y_BE = df["BE"].values
    y_pCO2 = df["pCO2"].values

    # Remove rows with missing age for risk computation
    valid_mask = df[PREDICTORS].notna().all(axis=1).values
    print(f"Risk computation on {valid_mask.sum()} patients "
          f"({(~valid_mask).sum()} excluded: missing predictors)")

    X_valid = X[valid_mask]
    y_pH_v  = y_pH[valid_mask]

    # Fit on all valid data (full-data conformal for risk scores)
    med_pH, lo_pH, hi_pH = fit_pi_model(X_valid, y_pH_v, X_valid)

    prob_720 = monte_carlo_risk(med_pH, lo_pH, hi_pH, threshold=7.20)
    prob_710 = monte_carlo_risk(med_pH, lo_pH, hi_pH, threshold=7.10)
    prob_700 = monte_carlo_risk(med_pH, lo_pH, hi_pH, threshold=7.00)

    bands = assign_risk_band(prob_720)

    # Build per-patient results
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    df_valid["pred_pH_median"]    = med_pH
    df_valid["pred_pH_lower_90"]  = lo_pH
    df_valid["pred_pH_upper_90"]  = hi_pH
    df_valid["P_pH_lt_720"]  = prob_720.round(4)
    df_valid["P_pH_lt_710"]  = prob_710.round(4)
    df_valid["P_pH_lt_700"]  = prob_700.round(4)
    df_valid["risk_band"]    = bands

    df_valid.to_csv(
        os.path.join(RESULTS_DIR, "patient_risk_scores.csv"), index=False)
    print("Per-patient risk scores saved.")

    # ── Table 6: Risk band distribution by Doppler group ───────────────────
    tab6_rows = []
    for grp, label in [(0, "Normal Doppler"), (1, "Abnormal Doppler")]:
        sub = df_valid[df_valid["group_doppler"] == grp]
        n   = len(sub)
        for band in ["Low", "Moderate", "High"]:
            count = (sub["risk_band"] == band).sum()
            tab6_rows.append({
                "Doppler_group": label,
                "Risk_band": band,
                "n": count,
                "pct": round(count / n * 100, 1) if n > 0 else 0,
            })

    tab6 = pd.DataFrame(tab6_rows)
    tab6.to_csv(os.path.join(RESULTS_DIR, "Tab6_risk_stratification.csv"),
                index=False)
    print("\nTable 6 — Risk band distribution:")
    print(tab6.to_string(index=False))

    # ── Summary statistics ──────────────────────────────────────────────────
    print("\nOverall risk band proportions:")
    for band in ["Low", "Moderate", "High"]:
        n = (df_valid["risk_band"] == band).sum()
        print(f"  {band}: n={n} ({n/len(df_valid)*100:.1f}%)")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
