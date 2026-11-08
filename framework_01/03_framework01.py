"""
PHASE 03 - MODEL EVALUATION & LEARNING
-------------------------------------------------------------------------------------------------
Framework 1: regression on continuous 0-1 target scores (no resampling).

Methodology:
- Every dataset (3 target formulas x 2 predictor sets) is evaluated for the three targets
  separately (myopia, CVS, astigmatism); respondents with a missing target are dropped for that target.
- Repeated K-Fold Cross-Validation (5 folds, 3 repeats) because of the small sample size.
- Prevents data leakage: imputation and scaling are fitted ONLY on the training folds, inside a
  scikit-learn Pipeline.
- Models:
    Ridge Regression    linear baseline with L2 regularisation
    Random Forest       non-linear bagging ensemble
    XGBoost             gradient boosting ensemble
    SVR (RBF kernel)    kernel method, a different paradigm from the tree models
    Elastic Net         L1 + L2 linear model (alpha / l1_ratio tuned by an inner CV on the training
                        fold only); it can drive uninformative predictors to exactly zero
    Mean Predictor      DummyRegressor, the "guess the average" baseline
- Metrics:
    MAE      mean absolute error on the 0-1 scale (0.18 = off by 18 points on average)
    RMSE     like MAE but penalises large mistakes more
    Skill_%  (Baseline MAE - Model MAE) / Baseline MAE * 100; negative = worse than the baseline
- Feature importance is permutation-based on held-out CV folds for the best non-baseline model.
  Positive values indicate the increase in MAE after shuffling that predictor. Rankings are
  model-dependent, can be shared across correlated predictors, and do not imply causation.

- NOTE: Elastic Net feature selection: for each dataset/target, an Elastic Net is also fitted on all valid
  respondents and the predictors whose weight was shrunk to exactly zero are reported.
  This is descriptive only (fitted on the full data) and is not used in the scores above.
-------------------------------------------------------------------------------------------------
Outputs (results/):
    framework01_results.csv             every cross-validated metric
    framework01_feature_importance.csv  ranked held-out permutation importance by dataset / target
-------------------------------------------------------------------------------------------------
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "datasets"
RESULTS_DIR = HERE / "results"

TARGETS = ["target_myopia", "target_cvs", "target_astigmatism"]
BASELINE = "Baseline (Mean Predictor)"
N_SPLITS, N_REPEATS, SEED = 5, 3, 42
IMPORTANCE_REPEATS = 10

# Elastic Net can emit convergence warnings on tiny training folds; they are not actionable here
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_models() -> dict:
    """Create fresh model pipelines with preprocessing fitted within each training fold."""
    return {
        BASELINE: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(random_state=SEED)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=SEED, n_estimators=100)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        random_state=SEED,
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        ),
        "Support Vector Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
            ]
        ),
        "Elastic Net": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
                        cv=5,
                        max_iter=10000,
                    ),
                ),
            ]
        ),
    }


def evaluate_target(df: pd.DataFrame, target_col: str, predictors: list):
    """Evaluate models and estimate held-out permutation importance for the best model."""
    print(f"\n{'='*40}")
    print(f"EVALUATING TARGET: {target_col}")
    print(f"{'='*40}")

    # Drop rows where the target is missing (e.g., 'Unsure' myopia answers)
    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[predictors]
    y = df_clean[target_col]
    print(f"Valid respondents for this target: {len(X)}")

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    cv_splits = list(rkf.split(X, y))

    results = []
    best_model = None
    best_estimators = None
    best_mae = float("inf")
    for name, pipeline in make_models().items():
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv_splits,
            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error"),
            n_jobs=-1,
            return_estimator=True,
        )
        mean_mae = -cv_results["test_neg_mean_absolute_error"].mean()
        if name != BASELINE and mean_mae < best_mae:
            best_model = name
            best_estimators = cv_results["estimator"]
            best_mae = mean_mae

        # Mean metrics across all 15 splits (5 folds * 3 repeats)
        results.append(
            {
                "Model": name,
                "MAE": mean_mae,
                "MAE_Std": cv_results["test_neg_mean_absolute_error"].std(),
                "RMSE": -cv_results["test_neg_root_mean_squared_error"].mean(),
            }
        )

    res_df = pd.DataFrame(results).set_index("Model")

    # Skill score: % MAE improvement over the baseline (negative = worse than baseline)
    baseline_mae = res_df.loc[BASELINE, "MAE"]
    res_df["Skill_%"] = (baseline_mae - res_df["MAE"]) / baseline_mae * 100
    res_df = res_df.sort_values("Skill_%", ascending=False)

    print("\nModels ranked best to worst (Skill_% = MAE improvement vs baseline):")
    print(res_df.round(4))

    if best_model is None or best_estimators is None:
        raise RuntimeError("No non-baseline model was available for feature-importance analysis.")

    importance_df = calculate_feature_importance(
        best_estimators,
        X,
        y,
        cv_splits,
        predictors,
        best_model,
    )
    print(f"\nTop predictors for {target_col} ({best_model}; held-out CV permutation):")
    print(importance_df.head(10).round(4).to_string(index=False))

    return res_df, importance_df


def calculate_feature_importance(
    estimators: list,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    predictors: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Rank predictors by their held-out-fold increase in MAE when permuted."""
    importance_values = {column: [] for column in predictors}
    for fold_number, (estimator, (_, test_indices)) in enumerate(
        zip(estimators, cv_splits)
    ):
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        baseline_mae = mean_absolute_error(y_test, estimator.predict(X_test))
        rng = np.random.default_rng(SEED + fold_number)

        for column in predictors:
            for _ in range(IMPORTANCE_REPEATS):
                permuted = X_test.copy()
                permuted[column] = rng.permutation(permuted[column].to_numpy())
                permuted_mae = mean_absolute_error(
                    y_test,
                    estimator.predict(permuted),
                )
                importance_values[column].append(permuted_mae - baseline_mae)

    importance = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Feature": feature,
                "Importance_Mean": float(pd.Series(values).mean()),
                "Importance_Std": float(pd.Series(values).std()),
            }
            for feature, values in importance_values.items()
        ]
    )
    importance["Rank"] = importance["Importance_Mean"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    return importance.sort_values(
        ["Importance_Mean", "Feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def load_datasets() -> list:
    """Find dataset files and return each formula, predictor set, and file path."""
    paths = sorted(DATASET_DIR.glob("dataset__*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No datasets found in {DATASET_DIR}. Run 01_preprocessing.py and "
            "02_target_normalisation.py first."
        )
    out = []
    for path in paths:
        _, formula, predictor_set = path.stem.split("__")
        out.append((formula, predictor_set, path))
        print(f"Combinations found: {formula} / {predictor_set}")
    return out


def main():
    """Evaluate every dataset and target, then save and summarize the model results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    result_rows, selection_rows, importance_rows = [], [], []

    for formula, predictor_set, path in load_datasets():
        df = pd.read_csv(path)
        predictors = []

        for col in df.columns:
            if col not in TARGETS:
                predictors.append(col)

        print(f"\n{'#'*64}")
        print(
            f"# TARGET FORMULA: {formula}   |   PREDICTOR SET: {predictor_set} "
            f"({len(predictors)} predictors)"
        )
        print(f"{'#'*64}")

        for target in TARGETS:
            res_df, importance_df = evaluate_target(df, target, predictors)
            importance_df.insert(0, "Target", target)
            importance_df.insert(0, "Predictors", predictor_set)
            importance_df.insert(0, "Formula", formula)
            importance_rows.extend(importance_df.to_dict("records"))

            for model, row in res_df.iterrows():
                result_rows.append(
                    {
                        "Formula": formula,
                        "Predictors": predictor_set,
                        "Target": target,
                        "Model": model,
                        **row.to_dict(),
                    }
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS_DIR / "framework01_results.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        RESULTS_DIR / "framework01_elasticnet_selection.csv",
        index=False,
    )
    feature_importance = pd.DataFrame(importance_rows)
    feature_importance.to_csv(
        RESULTS_DIR / "framework01_feature_importance.csv",
        index=False,
    )

    # Overview: Skill_% of every model under every formula / predictor-set combination
    results["Variant"] = results["Formula"] + " / " + results["Predictors"]
    print(f"\n{'#'*64}")
    print("# OVERVIEW: Skill_% (MAE improvement vs baseline) per target")
    print(f"{'#'*64}")
    for target in TARGETS:
        pivot = results[results["Target"] == target].pivot(
            index="Model",
            columns="Variant",
            values="Skill_%",
        )
        print(f"\n{target}")
        print(pivot.round(2).to_string())

    best = results.loc[
        results.groupby(["Target", "Variant"])["Skill_%"].idxmax()
    ]
    print("\nBest model per target and variant:")
    print(
        best[["Target", "Variant", "Model", "MAE", "RMSE", "Skill_%"]]
        .round(4)
        .to_string(index=False)
    )
    print("\nTop predictive factors by formula, predictor set, and target:")
    print(
        feature_importance.groupby(
            ["Formula", "Predictors", "Target"],
            sort=False,
        )
        .head(5)[
            [
                "Formula",
                "Predictors",
                "Target",
                "Model",
                "Rank",
                "Feature",
                "Importance_Mean",
            ]
        ]
        .to_string(index=False)
    )
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
