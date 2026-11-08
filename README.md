# Ocular Health Prediction Research

This repository contains two complementary machine-learning workflows for studying
relationships between digital-use patterns and self-reported vision outcomes. Both
frameworks use the survey data in `INPUT_ORG.csv`, but answer different prediction
questions:

| Framework | Task | Target representation | Resampling |
|---|---|---|---|
| Framework 01 | Regression | Continuous 0–1 myopia, CVS, and astigmatism scores | None |
| Framework 02 | Classification | Low, Moderate, and High classes derived from those scores | Compare no resampling with SMOTE-NC |

Neither framework establishes causation. Results depend on the quality and size of the
survey sample and should be interpreted as exploratory predictive evidence.

The generated directories and CSV files are ignored by Git. Run a framework's scripts
in order; each framework has its own preprocessing, dataset-generation, and evaluation
steps.



## Framework 01: continuous-score regression

```sh
python framework_01/01_preprocessing.py
python framework_01/02_target_normalisation.py
python framework_01/03_framework01.py
```

### Phase 1: preprocessing

`01_preprocessing.py` reads `INPUT_ORG.csv`, validates the expected Google Forms
headers, maps survey responses to ordinal codes, and creates audit files, distribution
summaries, and plots. It also reports non-empty responses that could not be mapped.
Review `framework_01/preprocessing_output/unmapped_categories.csv` before interpreting
model results.

Age and gender are retained for descriptive audit but excluded from modelling. The
model-candidate CSV includes the encoded predictors and the five outcome-question
inputs required to construct targets.

### Phase 2: target and predictor datasets

`02_target_normalisation.py` builds six datasets: three target formulas (`linear`,
`power`, `noisy_or`) crossed with two predictor presentations (`raw`, `combined`).
Target components are scaled using fixed theoretical survey ranges before aggregation.
The combined presentation replaces related predictors with five composite indices and
passes ungrouped predictors through unchanged. These fixed ranges and weights do not
learn from the data.

### Phase 3: regression

`03_framework01.py` evaluates myopia, CVS, and astigmatism scores separately using
repeated 5-fold cross-validation with 3 repeats. Preprocessing is fitted within each
training fold. It compares a mean-prediction baseline with Ridge Regression, Random
Forest, XGBoost, Support Vector Regression, and Elastic Net. The reported metrics are
MAE, RMSE, and MAE skill relative to the baseline.

For each dataset and target, the script also reports held-out-fold permutation
importance for the strongest non-baseline model. The importance table ranks individual
raw predictors or, for combined datasets, composite groups. Importance is the change
in held-out MAE after shuffling a feature; values can be negative, and rankings are
model-dependent, may be shared across correlated predictors, and do not imply
causation.

Key outputs in `framework_01/results/`:

- `framework01_results.csv` — cross-validated metrics for every model, target, formula,
  and predictor set.
- `framework01_feature_importance.csv` — ranked permutation importance, including the
  selected model and importance mean and standard deviation.

## Framework 02: risk-class classification
```sh
python framework_02/01_preprocessing.py
python framework_02/02_target_normalisation.py
python framework_02/03_framework02.py
```

The preprocessing and target/predictor dataset steps use the same survey definitions
and dataset variants as Framework 01, keeping the two analyses comparable. Framework
02 transforms each continuous 0–1 target score into Low, Moderate, or High using the
cutoffs in `framework_02/03_framework02.py` (default: below 1/3 is Low, below 2/3 is
Moderate, otherwise High).

The classifier compares a most-frequent baseline with Logistic Regression, Random
Forest, XGBoost, SVC, and LinearSVC. It uses repeated stratified 5-fold cross-validation
with 3 repeats and reports balanced accuracy, macro F1, and confusion matrices. Each
non-baseline model is evaluated both without resampling and with SMOTE-NC applied only
to the training fold. Imputation and scaling are also fitted within each fold.

At the end of each target evaluation, Framework 02 reports held-out-fold permutation
importance for the strongest non-baseline model/resampling combination. Raw datasets
show individual predictors; combined datasets show composite groups. Importance is
the decrease in held-out balanced accuracy after shuffling a feature. Values can be
negative, and rankings are model-dependent, may be shared across correlated
predictors, and do not imply causation.

Key outputs in `framework_02/results/`:

- `framework02_results.csv` — classification metrics for each model and resampling
  condition.
- `framework02_smote_effect.csv` — metric changes associated with SMOTE-NC.
- `framework02_feature_importance.csv` — ranked permutation importance by dataset and
  target.
- `confusion_matrices/` — figures comparing model predictions with the true classes.

## Interpreting the two frameworks

Framework 01 tests whether predictors estimate the continuous target scores more
accurately than a mean baseline. Framework 02 tests whether predictors distinguish
three score-based risk classes, including whether training-fold resampling changes
classification performance. The target classes in Framework 02 are derived from the
same underlying survey responses as the regression scores; they are not independent
clinical diagnoses.

Permutation importance describes how much a fitted model relies on a feature for its
held-out predictions. It is not a measure of the feature's causal effect, and
correlated features can dilute or share importance. Use it together with the
cross-validation metrics and the study's sample-size and self-report limitations.
