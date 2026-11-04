# Model Card — NSW Road Crash Severity Classifier

**Version:** 1.0.0  **Date:** 2026-08-24
**Best model:** Random Forest

## Test-set metrics (2024 holdout)
| Metric | Value |
|---|---|
| Macro-F1 | 0.5335 |
| Macro PR-AUC | 0.5486 |
| Macro ROC-AUC | 0.8261 |
| Fatal recall | 0.2685 |
| Fatal precision | 0.1314 |

## Full comparison
| Model                   |   Val Macro-F1 |   Test Macro-F1 |   Test Macro PR-AUC |   Test Macro ROC-AUC |   Fatal Recall |   Fatal Precision |   Fatal PR-AUC |   Injury F1 |   Non-casualty F1 |
|:------------------------|---------------:|----------------:|--------------------:|---------------------:|---------------:|------------------:|---------------:|------------:|------------------:|
| Random Forest           |         0.5271 |          0.5335 |              0.5486 |               0.8261 |         0.2685 |            0.1314 |         0.1244 |      0.7747 |            0.6493 |
| XGBoost                 |         0.5274 |          0.5251 |              0.5573 |               0.8307 |         0.396  |            0.1137 |         0.1287 |      0.7444 |            0.6541 |
| MLP (Entity Embeddings) |         0.5107 |          0.5137 |              0.5375 |               0.7994 |         0.2785 |            0.1092 |         0.0925 |      0.7419 |            0.6422 |
| Random Forest + SMOTE   |         0.4847 |          0.4971 |              0.5556 |               0.8317 |         0.0336 |            0.3125 |         0.1238 |      0.819  |            0.6117 |
| Logistic Regression     |         0.4693 |          0.4653 |              0.5357 |               0.803  |         0.7148 |            0.0651 |         0.1237 |      0.6282 |            0.6483 |
| Decision Tree           |         0.464  |          0.4566 |              0.4831 |               0.7092 |         0.4128 |            0.0558 |         0.0495 |      0.6705 |            0.601  |

## Training data
NSW Road Crash Data 2020-2024. Target: `Degree of crash` (Fatal/Injury/Non-casualty).
Split: train 2020-2022, validate 2023, test 2024 (time-based, untouched holdout).

## Excluded (outcome leakage)
['Degree of crash - detailed', 'No. killed', 'No. seriously injured', 'No. moderately injured', 'No. minor-other injured'], plus Crash ID / Reporting year / Identifying feature (admin/free-text),
and Year of crash as a raw feature (kept only as the split key).

## Known limitations
Fatal crashes are ~1.5% of the data; Fatal-class precision remains modest even with
class weighting / SMOTE. Correlational, not causal. Trained on 2020-2023, tested on 2024;
generalisation to other years/regions is unverified.

## Artifact
models/best_model.joblib (joblib sklearn/XGBoost Pipeline)

## Environment
python 3.13.15, torch 2.11.0+cpu, xgboost 3.4.1

## Second target added (21 September 2026)

The dashboard now also ships a logistic model fitted on **KSI** (killed or
seriously injured) — 20,977 crashes against 1,388 fatal. Same features, same
split, target swapped. Test AUC 0.631 against 0.732 for the fatal target:
serious injury is common enough (22.8% of crashes) that the road environment
separates it far less sharply than it separates death.

The artifact in this folder is still the fatal-target Random Forest from the
notebook. The two browser models live in `dashboard/data.js` and are regenerated
by `analysis/build_dashboard_data.py`.
