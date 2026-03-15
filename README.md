# Probabilistic Prediction of Umbilical Artery Acid–Base Status from Antenatal Doppler

Replication code for: *Probabilistic Prediction of Umbilical Artery Acid–Base Status from Antenatal Doppler in Preterm-Enriched High-Risk Pregnancies: A Machine Learning Approach with Calibrated Prediction Intervals and External Validation*

---

## Repository Structure

```
├── data/
│   ├── schema.md              # Variable definitions and coding
│   └── data_dictionary.csv    # Full data dictionary
├── scripts/
│   ├── 01_qc_harmonise.py     # Data quality control and harmonisation
│   ├── 02_eda.py              # Exploratory data analysis
│   ├── 03_model_development.py # Regression and classification models
│   ├── 04_conformal_pi.py     # Conformal prediction intervals
│   ├── 05_bootstrap_validation.py # Bootstrap internal validation
│   ├── 06_risk_engine.py      # Monte Carlo risk scores
│   ├── 07_dca.py              # Cross-validated Decision Curve Analysis
│   ├── 08_dml_evalue.py       # DML sensitivity analysis (Additional file 1)
│   ├── 09_external_validation.py # External cohort validation
│   └── 10_figures.py          # All manuscript figures
├── figures/                   # Output figures (300 DPI)
├── results/                   # Output tables (CSV)
└── requirements.txt
```

---

## Data

The de-identified analysis dataset is available in `data/` as CSV files:

- `clean_primary.csv` — development cohort (N=222; Soft-cleaned primary dataset)
- `clean_strict.csv` — development cohort (N=222; Strict-cleaned; identical to primary)


**Variable names** are defined in `data/schema.md`.

The source Excel files (`normal-cases.xlsx`, `abnormal-cases.xlsx`) contained raw data as received; the `01_qc_harmonise.py` script reproduces the cleaning steps and produces the CSV files above.

---

## Reproducibility

All analyses use a fixed random seed (`SEED = 42`). Python version: 3.11.

```bash
pip install -r requirements.txt
python scripts/01_qc_harmonise.py
python scripts/02_eda.py
python scripts/03_model_development.py
python scripts/04_conformal_pi.py
python scripts/05_bootstrap_validation.py
python scripts/06_risk_engine.py
python scripts/07_dca.py
python scripts/08_dml_evalue.py
python scripts/09_external_validation.py
python scripts/10_figures.py
```

Each script saves its outputs to `results/` and `figures/`. Scripts are designed to run sequentially; intermediate outputs from earlier scripts are loaded by later ones.

---

## Key Dependencies

See `requirements.txt`. Core packages: scikit-learn 1.4.0, numpy 1.26, pandas 2.2, scipy 1.12, matplotlib 3.8, seaborn 0.13.
