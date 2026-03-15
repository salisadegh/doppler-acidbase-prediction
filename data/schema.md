# Data Schema

## Development Cohort (`clean_primary.csv`, `clean_strict.csv`)

| Variable       | Type    | Units     | Description                                              |
|---------------|---------|-----------|----------------------------------------------------------|
| `GA`          | float   | weeks     | Gestational age at Doppler examination                   |
| `age`         | float   | years     | Maternal age                                             |
| `UA_PI`       | float   | —         | Umbilical artery pulsatility index                       |
| `pH`          | float   | —         | Umbilical artery blood gas pH at delivery                |
| `pCO2`        | float   | mmHg      | Umbilical artery pCO₂ at delivery                       |
| `BE`          | float   | mEq/L     | Umbilical artery base excess at delivery                 |
| `group_doppler` | int   | 0/1       | 0 = normal Doppler; 1 = abnormal Doppler                 |
| `acidemia_720` | int    | 0/1       | 1 if pH < 7.20 (primary binary endpoint)                 |
| `acidemia_710` | int    | 0/1       | 1 if pH < 7.10 (zero events; deferred)                  |
| `flag_any`    | int     | 0/1       | 1 if any variable outside plausibility range             |

## External Validation Cohort (`external_validation.csv`)

Same variable structure as development cohort. Source: independent private antenatal clinic.

## Plausibility Thresholds Applied During QC

| Variable | Min  | Max  |
|----------|------|------|
| GA       | 20   | 45   |
| age      | 12   | 55   |
| pH       | 6.80 | 7.60 |
| pCO2     | 10   | 90   |
| BE       | −30  | +20  |
| UA_PI    | 0.20 | 3.00 |

## Missing Data

One normal-Doppler record had missing `age`. This case is included in all analyses; `age` was imputed within each cross-validation fold using the training-fold median to prevent data leakage. The case is excluded from risk-score computation (requires all three predictors for Monte Carlo sampling).
