# inail-severity-benchmark

Code for the article **"Using Italian open data and machine learning to characterise regional
patterns in workplace accident outcomes"** (C. D'Urso).

The pipeline estimates how far the severity of an indemnified workplace accident (type of
INAIL compensation: temporary disability TE, lump sum CA, direct annuity RD, survivors'
annuity RS) can be predicted and described from three administrative variables (age, gender,
economic sector) in Italy's four macro-regions, using INAIL open data for 2020-2024.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 or later
pip install -r requirements.txt
# download the four INAIL CSV files (Zenodo, see data/README.md) into data/raw/
python run.py all                  # complete pipeline (several hours, see below)
python run.py all --skip-slow      # main results only (about 1 hour)
```

To check that everything runs without the real data:

```bash
python tests/smoke_test.py         # synthetic data, about 15 minutes
```

All settings (paths, seeds, split proportions, bootstrap replicates, tuning budget) are in
`config.yaml`. Results are written to `outputs/`.

## Steps and correspondence with the article

| Step (`python run.py <step>`) | Output | Article |
|---|---|---|
| `prepare` | `data/*.pkl`, `tables/Table1_records_removed.csv` | Table 1 |
| `descriptives` | `tables/Appendix*.csv`, `figures/Figure6-9_*.png` | Appendices B and C, Figures 6-9 |
| `predictive` | `predictive/res_<region>.json` | Tables 3 and 4 |
| `inference` | `inference/*.csv`, `tables/Table5_profiles.csv` | Table 2, Table 5, Tables S1-S3 |
| `pooled` | `inference/pooled_lrt.json` | Regional differences, Table S4 |
| `selection` | `supplementary/S5_selection_stages.csv` | Table S5 |
| `ggt` | `supplementary/S6_*.csv` | Table S6 |
| `bootstrap` | `inference/bootstrap_<region>.csv` | Table S7 |
| `tuning` | `tuning/tuned_<region>.json` | Table S8 |
| `chrono` | `chronological/chrono_<region>.json` | Table S9 |
| `stability` | `stability/stability_<region>.json` | Stability of the model (SHAP) |
| `figures` | `figures/Figure4*, Figure5*, Figure10*, Figure11*` | Figures 4, 5, 10, 11 |
| `report` | `tables/*.csv`, `supplementary/Supplementary_Tables.xlsx`, `environment.json` | Tables 2-5, S1-S9 |

Steps that work by macro-region accept an optional region, e.g. `python run.py tuning South`.

Measured running times on a 2-core machine with 8 GB of memory: `predictive` 3-5 min per
region, `inference` about 15 min, `pooled`, `selection` and `ggt` a few minutes, `chrono` 2-5 min
per region, `bootstrap` (200 replicates) 17-29 min per region, `stability` (20 runs) 10-13 min
per region, `tuning` (20 configurations per model) 30-40 min per region; about 5 hours in total.
The folder `results/` contains the outputs of the run used for the article (tables,
supplementary workbook, figures and `environment.json`), for comparison.

## Methods implemented

**Data and selection** (`data.py`). Records are removed sequentially if the economic sector,
ATECO code, age or place of occurrence is undetermined, if the claim was
not indemnified (Indennizzo = NE), if the accident occurred while commuting (in itinere), if
the employer insurance position is undetermined, or if the worker was under 15. Records
with an undetermined Large Tariff Group (GGT), a variable not used by the models, are kept;
setting `selection: exclude_missing_ggt: true` reproduces the earlier sample (Table S6).

**Features** (`features.py`). Predictive models: gender, age in years, one-hot ATECO sections
B-U (A omitted). Inferential models: gender, age centred at 45 per ten years, ATECO sections
with G (trade) as reference and T, U merged into S.

**Split.** Stratified 60/20/20 train/validation/test split (seed 42). The validation set is
used only for early stopping and for selecting decision thresholds; the test set only for
the final evaluation.

**Predictive models** (`models.py`, `predictive.py`), with the configurations used in the article:

| Model | Configuration | Imbalance | Decision rule |
|---|---|---|---|
| Logistic regression (four classes) | scikit-learn `LogisticRegression`, lbfgs, L2, C = 1, max_iter 2000 | `class_weight="balanced"` | argmax; binary task: Youden threshold on P(TE) |
| XGBoost (four classes) | library defaults: 100 trees, depth 6, learning rate 0.3; `multi:softprob` | inverse-frequency sample weights | argmax; binary task: Youden threshold on P(TE) |
| Balanced random forest (four classes) | imbalanced-learn, 500 trees, max depth 12, min split 5, min leaf 2, sqrt features | under-sampling to the minority class | class-specific thresholds (0.30-0.70, step 0.10, 625 combinations) maximising balanced accuracy on validation; binary task: Youden threshold |
| CatBoost (binary, TE vs other) | 1,000 iterations, learning rate 0.01, depth 8, L2 = 5, early stopping 200 on validation (AUC) | `auto_class_weights="Balanced"` | Youden threshold on validation |

Unweighted probability models (Table 4, Panel C): class-frequency model, unweighted
logistic regression, unweighted XGBoost, unweighted binary CatBoost and a saturated cell model
(every age x gender x ATECO cell, additive smoothing towards the class frequencies), compared
by log-loss, McFadden pseudo-R2, Brier score, Brier skill score and ROC-AUC.

**Uncertainty.** Test-set metrics: 95% percentile bootstrap intervals, 1,000 resamples of the
test set (300 for AUC), seed 2026. Logit coefficients: cluster-robust standard errors by
worker (`IdentificativoInfortunato`); Table S7 compares them with a worker-level cluster
bootstrap (200 replicates, seed 2027).

**Inference** (`inference.py`, `pooled.py`). Unpenalised multinomial logit (statsmodels
`MNLogit`, reference outcome TE) with Benjamini-Hochberg correction (FDR 0.05) across all
coefficients of each regional model; proportional-odds ordinal logit (`OrderedModel`); binary
logit by period with and without year fixed effects; pooled binary logit with likelihood-ratio
tests of regional differences. Coefficients of outcome-sector cells without any case are
flagged as not estimable (n.e.).

**Comparable tuning** (`tuning.py`, Table S8). Every model receives the same budget (20
random configurations from the search spaces in `models.SPACES`), fitted on the training set
and selected on the validation set by a threshold-independent criterion (macro AUC, binary
AUC, or log-loss for the unweighted models); the selected configuration is evaluated on the
test set exactly as in Table 4.

**Chronological validation** (Table S9). Training on 2020-2022, threshold selection and
early stopping on 2023, test on 2024.

## Reproducibility

Seeds are fixed in `config.yaml`; library versions are pinned in `requirements.txt` and the
versions actually used are recorded in `outputs/environment.json`. Results may differ in the
last decimal with other library versions or numbers of threads.

## Licence

Code: MIT (see `LICENSE`). Data: INAIL open data, CC BY 4.0.

## Version history

- **1.1.0** (November 2026). Table 1 separates the claims of the Agriculture and State insurance
  managements (excluded by design, no economic sector) from the claims of the Industry and Services
  management with a missing economic sector, and the removal of invalid ages from that of workers
  under 15; Supplementary Table S5 adds the profile of the otherwise eligible records lost because
  the economic sector is missing; Figure 5 no longer shows minor axis labels in scientific notation.
  The analysed records and all estimates are unchanged.
- **1.0.0** (November 2026). Version used for the first revision of the article.
