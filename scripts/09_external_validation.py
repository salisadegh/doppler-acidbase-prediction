"""
09_external_validation.py
-------------------------
Applies the locked development-cohort model to the external validation cohort.
Locked model: no updating, no recalibration, no predictor re-selection.
Produces Table 9.

Usage:
    python scripts/09_external_validation.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
import os

SEED = 42
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTORS    = ["GA", "age", "UA_PI"]
OUTCOMES_REG  = ["pH", "BE", "pCO2"]
OUTCOME_CLASS = "acidemia_720"
NOMINAL       = 0.90


def load_dev():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def load_ext():
    df = pd.read_csv(os.path.join(DATA_DIR, "external_validation.csv"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def train_locked_ridge(X_train, y_train):
    """Locked Ridge regression model trained on full development cohort."""
    m = Pipeline([("sc", StandardScaler()),
                  ("r", Ridge(alpha=1.0, random_state=SEED))])
    m.fit(X_train, y_train)
    return m


def train_locked_rf(X_train, y_train):
    """Locked Random Forest classifier trained on full development cohort."""
    m = Pipeline([("sc", StandardScaler()),
                  ("rf", RandomForestClassifier(
                      n_estimators=200, class_weight="balanced",
                      random_state=SEED, n_jobs=-1))])
    m.fit(X_train, y_train)
    return m


def conformal_pi_coverage(X_train, y_train, X_ext, y_ext,
                           nominal=NOMINAL, seed=SEED):
    """
    Fit conformal PI on training data, evaluate coverage on external cohort.
    Uses the same calibration fraction as development.
    """
    np.random.seed(seed)
    n_calib = max(10, int(len(X_train) * 0.25))
    calib_mask = np.zeros(len(X_train), dtype=bool)
    calib_mask[np.random.choice(len(X_train), n_calib, replace=False)] = True
    fit_mask = ~calib_mask

    sc = StandardScaler()
    X_fit_sc   = sc.fit_transform(X_train[fit_mask])
    X_calib_sc = sc.transform(X_train[calib_mask])
    X_ext_sc   = sc.transform(X_ext)

    lower_q = (1 - nominal) / 2
    upper_q = 1 - lower_q

    qr_lo  = QuantileRegressor(quantile=lower_q,  alpha=0.10,
                               solver="highs").fit(X_fit_sc, y_train[fit_mask])
    qr_hi  = QuantileRegressor(quantile=upper_q,  alpha=0.10,
                               solver="highs").fit(X_fit_sc, y_train[fit_mask])

    lo_cal = qr_lo.predict(X_calib_sc)
    hi_cal = qr_hi.predict(X_calib_sc)
    scores = np.maximum(lo_cal - y_train[calib_mask],
                        y_train[calib_mask] - hi_cal)
    correction = np.quantile(scores, nominal)

    lo_ext = qr_lo.predict(X_ext_sc) - correction
    hi_ext = qr_hi.predict(X_ext_sc) + correction

    coverage = np.mean((y_ext >= lo_ext) & (y_ext <= hi_ext))
    width    = np.mean(hi_ext - lo_ext)
    return coverage, width


def calibration_metrics(y_true, y_prob):
    """Calibration-in-the-large and calibration slope."""
    from scipy.special import logit
    prev = y_true.mean()
    citl = np.log(y_prob.mean() / (1 - y_prob.mean())) - np.log(prev / (1 - prev))

    # Calibration slope via logistic regression on logit predicted
    logit_pred = np.log(np.clip(y_prob, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(y_prob, 1e-6, 1 - 1e-6)))
    from sklearn.linear_model import LogisticRegression
    lr_cal = LogisticRegression(fit_intercept=True, max_iter=1000)
    lr_cal.fit(logit_pred.reshape(-1, 1), y_true)
    slope = lr_cal.coef_[0][0]
    return round(citl, 3), round(slope, 3)


def main():
    df_dev = load_dev()
    df_ext = load_ext()

    X_dev = df_dev[PREDICTORS].values
    X_ext = df_ext[PREDICTORS].values
    y_cls_dev = df_dev[OUTCOME_CLASS].values
    y_cls_ext = df_ext[OUTCOME_CLASS].values

    print(f"Development cohort: N={len(df_dev)}, "
          f"acidemia: {y_cls_dev.sum()} ({y_cls_dev.mean()*100:.1f}%)")
    print(f"External cohort:    N={len(df_ext)}, "
          f"acidemia: {y_cls_ext.sum()} ({y_cls_ext.mean()*100:.1f}%)")

    rows = []

    # ── Regression ──────────────────────────────────────────────────────────
    for out in OUTCOMES_REG:
        y_dev = df_dev[out].values
        y_ext = df_ext[out].values

        model = train_locked_ridge(X_dev, y_dev)
        pred_ext = model.predict(X_ext)

        mae  = mean_absolute_error(y_ext, pred_ext)
        rmse = np.sqrt(mean_squared_error(y_ext, pred_ext))
        r2   = r2_score(y_ext, pred_ext)

        rows.append({
            "Metric": f"{out} — R²",
            "Dev_CV": "-",
            "External": round(r2, 3),
        })
        rows.append({
            "Metric": f"{out} — MAE",
            "Dev_CV": "-",
            "External": round(mae, 4),
        })
        rows.append({
            "Metric": f"{out} — RMSE",
            "Dev_CV": "-",
            "External": round(rmse, 4),
        })

        if out == "pH":
            cov_90, _ = conformal_pi_coverage(X_dev, y_dev, X_ext, y_ext, 0.90)
            cov_95, _ = conformal_pi_coverage(X_dev, y_dev, X_ext, y_ext, 0.95)
            rows.append({
                "Metric": "pH — PI coverage @90%",
                "Dev_CV": 0.886,
                "External": round(cov_90, 3),
            })
            rows.append({
                "Metric": "pH — PI coverage @95%",
                "Dev_CV": 0.932,
                "External": round(cov_95, 3),
            })

    # ── Classification (RF — deployment model) ─────────────────────────────
    rf_model = train_locked_rf(X_dev, y_cls_dev)
    probs_ext = rf_model.predict_proba(X_ext)[:, 1]

    auroc = roc_auc_score(y_cls_ext, probs_ext)
    auprc = average_precision_score(y_cls_ext, probs_ext)
    brier = brier_score_loss(y_cls_ext, probs_ext)
    citl, slope = calibration_metrics(y_cls_ext, probs_ext)

    for label, val in [
        ("Acidemia AUROC — RF", round(auroc, 3)),
        ("Acidemia AUPRC — RF", round(auprc, 3)),
        ("Brier score — RF",    round(brier, 3)),
        ("Calibration-in-the-large", round(citl, 3)),
        ("Calibration slope",        round(slope, 3)),
    ]:
        rows.append({"Metric": label, "Dev_CV": "-", "External": val})

    tab9 = pd.DataFrame(rows)
    tab9.to_csv(os.path.join(RESULTS_DIR, "Tab9_external_validation.csv"),
                index=False)
    print("\nTable 9 — External validation:")
    print(tab9.to_string(index=False))

    # ── Subgroup analysis (term vs preterm) ────────────────────────────────
    print("\nSubgroup analysis by gestational age:")
    for label, mask in [
        ("Preterm (GA<37w)", df_ext["GA"] < 37),
        ("Term (GA≥37w)",    df_ext["GA"] >= 37),
    ]:
        if mask.sum() < 5:
            print(f"  {label}: n={mask.sum()} — too few for subgroup analysis")
            continue
        y_sub = y_cls_ext[mask.values]
        p_sub = probs_ext[mask.values]
        if y_sub.sum() < 2:
            print(f"  {label}: n={mask.sum()}, events={y_sub.sum()} "
                  f"— too few events for classification metrics")
            continue
        auc_sub = roc_auc_score(y_sub, p_sub)
        print(f"  {label}: n={mask.sum()}, events={y_sub.sum()}, "
              f"AUROC={auc_sub:.3f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
