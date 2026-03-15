"""
03_model_development.py
-----------------------
Regression and classification model development with nested cross-validation.
Produces Tables 2, 3, and 6b from the manuscript.

Usage:
    python scripts/03_model_development.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
)
import os

SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTCOMES_REG   = ["pH", "BE", "pCO2"]
OUTCOME_CLASS  = "acidemia_720"
PREDICTORS_CLIN  = ["GA", "age"]
PREDICTORS_FULL  = ["GA", "age", "UA_PI"]
OUTER_FOLDS = 5
INNER_FOLDS = 3


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_primary.csv"))
    # Impute missing age with median (full dataset median for overview;
    # within-fold for CV — handled below)
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.sum() / len(y_true) * abs(acc - conf)
    return ece_val


def make_reg_model(name, preds):
    if name == "null":
        return None
    if name in ("clinical", "doppler"):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=SEED)),
        ])
    if name == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=200, random_state=SEED, n_jobs=-1)),
        ])
    raise ValueError(f"Unknown model: {name}")


def make_clf_model(name):
    if name == "null":
        return None
    if name in ("clinical", "doppler"):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced",
                solver="lbfgs", max_iter=1000, random_state=SEED)),
        ])
    if name == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=SEED, n_jobs=-1)),
        ])
    raise ValueError(f"Unknown model: {name}")


def cv_regression(df, outcome, predictors, model_obj, n_splits=OUTER_FOLDS):
    """Nested CV regression — returns metrics dict."""
    X = df[predictors].values
    y = df[outcome].values

    if model_obj is None:
        # Null model: predict training mean
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SEED)
        strat = df[OUTCOME_CLASS].values
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in outer_cv.split(X, strat):
            preds[test_idx] = y[train_idx].mean()
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=SEED)
        strat = df[OUTCOME_CLASS].values
        preds = cross_val_predict(model_obj, X, y, cv=outer_cv)

    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def cv_classification(df, outcome, predictors, model_obj, n_splits=OUTER_FOLDS):
    """Nested CV classification — returns metrics dict."""
    X = df[predictors].values
    y = df[outcome].values

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=SEED)

    if model_obj is None:
        prev = y.mean()
        probs = np.full(len(y), prev)
    else:
        probs = cross_val_predict(model_obj, X, y, cv=outer_cv,
                                  method="predict_proba")[:, 1]

    auroc  = roc_auc_score(y, probs)
    auprc  = average_precision_score(y, probs)
    brier  = brier_score_loss(y, probs)
    ece_v  = ece(y, probs)
    return {
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "Brier": round(brier, 4),
        "ECE":   round(ece_v, 4),
    }


def main():
    df = load_data()
    print(f"Loaded: N={len(df)}, acidemia events: {df[OUTCOME_CLASS].sum()}")

    # ── Regression ─────────────────────────────────────────────────────────
    reg_rows = []
    configs = [
        ("Null",          "null",     PREDICTORS_FULL),
        ("Clinical",      "clinical", PREDICTORS_CLIN),
        ("Doppler-added", "doppler",  PREDICTORS_FULL),
        ("RandomForest",  "rf",       PREDICTORS_FULL),
    ]

    for outcome in OUTCOMES_REG:
        for label, mname, preds in configs:
            model = make_reg_model(mname, preds)
            m = cv_regression(df, outcome, preds, model)
            reg_rows.append({
                "Outcome": outcome, "Model": label,
                **m
            })

    tab_reg = pd.DataFrame(reg_rows)
    tab_reg.to_csv(os.path.join(RESULTS_DIR, "Tab2_regression_performance.csv"),
                   index=False)
    print("\nTable 2 — Regression performance:")
    print(tab_reg.to_string(index=False))

    # ── Classification ──────────────────────────────────────────────────────
    clf_rows = []
    for label, mname, preds in configs:
        model = make_clf_model(mname)
        m = cv_classification(df, OUTCOME_CLASS, preds, model)
        clf_rows.append({"Model": label, **m})

    tab_clf = pd.DataFrame(clf_rows)
    tab_clf.to_csv(os.path.join(RESULTS_DIR, "Tab3_classification_performance.csv"),
                   index=False)
    print("\nTable 3 — Classification performance:")
    print(tab_clf.to_string(index=False))

    # ── Added value summary ─────────────────────────────────────────────────
    av_rows = []
    for outcome in OUTCOMES_REG:
        clin = tab_reg[(tab_reg.Outcome == outcome) &
                       (tab_reg.Model == "Clinical")].iloc[0]
        dopp = tab_reg[(tab_reg.Outcome == outcome) &
                       (tab_reg.Model == "Doppler-added")].iloc[0]
        av_rows.append({
            "Outcome": outcome,
            "Clinical_R2": clin.R2, "Doppler_R2": dopp.R2,
            "Delta_R2": round(dopp.R2 - clin.R2, 3),
            "Clinical_RMSE": clin.RMSE, "Doppler_RMSE": dopp.RMSE,
            "Delta_RMSE": round(dopp.RMSE - clin.RMSE, 4),
        })
    tab_av = pd.DataFrame(av_rows)
    tab_av.to_csv(os.path.join(RESULTS_DIR, "Tab_AddedValue.csv"), index=False)

    print("\nAdded value of UA/PI:")
    print(tab_av.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
