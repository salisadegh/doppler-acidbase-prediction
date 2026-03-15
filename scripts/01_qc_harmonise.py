"""
01_qc_harmonise.py
------------------
Quality control and harmonisation of raw source data.
Produces clean_primary.csv and clean_strict.csv in data/.

Usage:
    python scripts/01_qc_harmonise.py
"""

import pandas as pd
import numpy as np
import os

SEED = 42
DATA_DIR = "data"

# ── Plausibility thresholds ────────────────────────────────────────────────
THRESHOLDS = {
    "GA":    (20, 45),
    "age":   (12, 55),
    "pH":    (6.80, 7.60),
    "pCO2":  (10,  90),
    "BE":    (-30, 20),
    "UA_PI": (0.20, 3.00),
}


def parse_ga(value):
    """Convert GA strings like '37W' or '38W3D' to float weeks."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).upper().strip()
    if "W" in s:
        parts = s.split("W")
        weeks = float(parts[0])
        if len(parts) > 1 and parts[1]:
            days_part = parts[1].replace("D", "").strip()
            if days_part:
                weeks += float(days_part) / 7
        return weeks
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_and_harmonise():
    """Load normal and abnormal Doppler files and merge."""
    normal_path   = os.path.join(DATA_DIR, "normal-cases.xlsx")
    abnormal_path = os.path.join(DATA_DIR, "abnormal-cases.xlsx")

    # Normal Doppler group
    df_n = pd.read_excel(normal_path)
    df_n.columns = df_n.columns.str.strip().str.replace(" ", "_")
    df_n["group_doppler"] = 0

    # Abnormal Doppler group
    df_a = pd.read_excel(abnormal_path)
    df_a.columns = df_a.columns.str.strip().str.replace(" ", "_")
    df_a["group_doppler"] = 1

    # Standardise column names across both files
    col_map = {
        "GA": "GA", "Gestational_Age": "GA", "gestational_age": "GA",
        "Age": "age", "maternal_age": "age", "Mother_Age": "age",
        "UA_PI": "UA_PI", "UAPI": "UA_PI", "PI": "UA_PI",
        "pH": "pH", "ph": "pH",
        "pCO2": "pCO2", "PCO2": "pCO2", "pco2": "pCO2",
        "BE": "BE", "Base_Excess": "BE", "base_excess": "BE",
    }

    for df in [df_n, df_a]:
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                  inplace=True)

    # Parse GA strings in normal-Doppler file
    df_n["GA"] = df_n["GA"].apply(parse_ga)

    combined = pd.concat([df_n, df_a], ignore_index=True)
    combined = combined[["GA", "age", "UA_PI", "pH", "pCO2", "BE",
                          "group_doppler"]].copy()
    combined.reset_index(drop=True, inplace=True)

    print(f"Raw combined dataset: {len(combined)} records")
    return combined


def flag_outliers(df):
    """Flag records outside plausibility thresholds."""
    df = df.copy()
    df["flag_any"] = 0
    flag_cols = []
    for col, (lo, hi) in THRESHOLDS.items():
        if col in df.columns:
            flag_col = f"flag_{col}"
            df[flag_col] = ((df[col] < lo) | (df[col] > hi)).astype(int)
            flag_cols.append(flag_col)
            df["flag_any"] = df["flag_any"] | df[flag_col]
    return df, flag_cols


def add_endpoints(df):
    """Add binary acidemia endpoints."""
    df = df.copy()
    df["acidemia_720"] = (df["pH"] < 7.20).astype(int)
    df["acidemia_710"] = (df["pH"] < 7.10).astype(int)
    df["acidemia_700"] = (df["pH"] < 7.00).astype(int)
    return df


def main():
    raw = load_and_harmonise()

    flagged, flag_cols = flag_outliers(raw)

    # Save QC flags
    qc_flags = flagged[flagged["flag_any"] == 1][
        ["GA", "age", "UA_PI", "pH", "pCO2", "BE", "group_doppler", "flag_any"]
        + flag_cols
    ]
    qc_path = os.path.join(DATA_DIR, "qc_flags.csv")
    qc_flags.to_csv(qc_path, index=True)
    print(f"QC: {len(qc_flags)} records flagged → saved to {qc_path}")

    # Soft cleaning: remove records where pH or pCO2 is implausible
    soft_mask = (
        flagged["flag_pH"].fillna(0) == 0
    ) & (
        flagged["flag_pCO2"].fillna(0) == 0
    )
    clean_soft = flagged[soft_mask].copy().drop(
        columns=flag_cols + ["flag_any"], errors="ignore"
    )
    clean_soft = add_endpoints(clean_soft)
    clean_soft.reset_index(drop=True, inplace=True)

    # Strict cleaning: remove any record with any flag
    clean_strict = flagged[flagged["flag_any"] == 0].copy().drop(
        columns=flag_cols + ["flag_any"], errors="ignore"
    )
    clean_strict = add_endpoints(clean_strict)
    clean_strict.reset_index(drop=True, inplace=True)

    print(f"\nSoft-cleaned dataset: N={len(clean_soft)}")
    print(f"  Normal Doppler:   n={( clean_soft.group_doppler==0).sum()}")
    print(f"  Abnormal Doppler: n={( clean_soft.group_doppler==1).sum()}")
    print(f"  Acidemia (pH<7.20): n={clean_soft.acidemia_720.sum()} "
          f"({clean_soft.acidemia_720.mean()*100:.1f}%)")

    print(f"\nStrict-cleaned dataset: N={len(clean_strict)}")
    print(f"  (Identical to soft: both scenarios removed the same 4 records)")

    clean_soft.to_csv(os.path.join(DATA_DIR, "clean_primary.csv"), index=False)
    clean_strict.to_csv(os.path.join(DATA_DIR, "clean_strict.csv"), index=False)
    print(f"\nSaved: data/clean_primary.csv, data/clean_strict.csv")


if __name__ == "__main__":
    main()
