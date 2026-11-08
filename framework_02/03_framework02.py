"""
PHASE 03 - MODEL EVALUATION & LEARNING
-------------------------------------------------------------------------------------------------
Framework 2: classification (Low / Moderate / High) with and without SMOTE-NC oversampling.

Methodology:
- The continuous 0-1 target scores are cut into three classes with CLASS_CUTOFFS
  (default: < 1/3 Low, < 2/3 Moderate, otherwise High). Edit the cutoffs below to change them.
- Every dataset (3 target formulas x 2 predictor sets) is evaluated for the three targets separately.
- Repeated Stratified K-Fold (5 folds, 3 repeats), so every fold keeps the class proportions.
- Every model is run twice: WITHOUT resampling and WITH SMOTE-NC on the training fold only. SMOTE-NC
  (imblearn) interpolates continuous features and picks the majority category among the neighbours for
  categorical ones, so synthetic respondents keep valid answer codes. The test fold is never resampled.
- Prevents data leakage: per fold the imputer, SMOTE-NC and scaler are fitted on the training fold only
  (an imblearn Pipeline applies the sampler during fit and skips it during predict).
- Feature types for SMOTE-NC (CATEGORICAL_COLUMNS): the discrete survey answer codes are categorical.
  `age_first_rx` and the composite indices of the "combined" predictor set are continuous.
- Models:
    Logistic Regression   linear baseline, L2 penalty (the scikit-learn default)
    Random Forest         non-linear bagging ensemble
    XGBoost               gradient boosting ensemble
    SVC (RBF kernel)      kernel method
    LinearSVC (L1)        sparse linear model; the L1 penalty zeroes uninformative predictors
    Most-Frequent         DummyClassifier baseline (always predicts the largest class)
- Metrics (both are robust to class imbalance):
    Balanced Accuracy     mean recall over the classes, so rare high-risk respondents count as much as
                          common low-risk ones
    Macro F1              F1 averaged over the classes with equal weight; punishes ignoring minorities
    Confusion matrices    summed over all test folds and shown as average respondents per repeat
- Feature importance is permutation-based on held-out CV folds for the best non-baseline
  model/resampling combination. Positive values indicate a balanced-accuracy decrease
  after shuffling that predictor. Rankings are model-dependent, can be shared across
  correlated predictors, and do not imply causation.
- Small classes: SMOTE-NC needs at least 2 respondents of a class in the training fold, and the number of
  neighbours is lowered automatically for very small classes. Classes with a single respondent are left
  as they are.

Outputs (results/):
    framework02_results.csv              every metric for every dataset / target / model / resampling
    framework02_smote_effect.csv         per model: balanced-accuracy change caused by SMOTE-NC
    framework02_feature_importance.csv   ranked held-out permutation importance by dataset / target
    confusion_matrices/*.png             one figure per dataset and target
-------------------------------------------------------------------------------------------------
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "datasets"
RESULTS_DIR = HERE / "results"
CM_DIR = RESULTS_DIR / "confusion_matrices"

TARGETS = ["target_myopia", "target_cvs", "target_astigmatism"]
N_SPLITS, N_REPEATS, SEED = 5, 3, 42
IMPORTANCE_REPEATS = 10

# Score < first cutoff -> Low, < second -> Moderate, otherwise High
CLASS_CUTOFFS = (1 / 3, 2 / 3)
CLASS_NAMES = ["Low", "Moderate", "High"]

# Discrete survey answer codes (treated as categorical by SMOTE-NC). Every other column,
# i.e. age_first_rx and the composite indices of the "combined" set, is continuous.
CATEGORICAL_COLUMNS = {
    "daily_screen_time",
    "continuous_usage",
    "intensity",
    "lighting",
    "multi_device",
    "phone_distance",
    "monitor_distance",
    "blue_light_filter",
    "before_bed_usage",
    "profession",
    "outdoor_activity",
    "genetics",
    "has_prescription",
}
SMOTE_K = 5

BASELINE = "Baseline (Most Frequent)"
NO_RESAMPLING, WITH_SMOTE = "None", "SMOTE-NC"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# model name -> (unfitted estimator, needs feature scaling)
MODELS = {
    BASELINE: (DummyClassifier(strategy="most_frequent"), False),
    "Logistic Regression (L2)": (LogisticRegression(C=1.0, max_iter=2000), True),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=SEED), False),
    "XGBoost": (
        XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        False,
    ),
    "SVC (RBF)": (SVC(kernel="rbf", C=1.0), True),
    "LinearSVC (L1)": (
        LinearSVC(
            penalty="l1",
            dual=False,
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        ),
        True,
    ),
}


def to_classes(score: pd.Series) -> pd.Series:
    """Convert continuous scores to Low, Moderate, or High integer class labels."""
    cutoffs = np.round(CLASS_CUTOFFS, 6)
    return pd.Series(np.digitize(np.round(score.to_numpy(), 6), cutoffs), index=score.index)


def make_sampler(y_encoded: np.ndarray, n_categorical: int, seed: int):
    """Create a SMOTE sampler for underrepresented classes with at least two samples."""
    counts = np.bincount(y_encoded)
    majority = counts.max()
    eligible = {c: int(majority) for c, n in enumerate(counts) if 2 <= n < majority}
    if not eligible:
        return None
    k = max(1, min(SMOTE_K, min(counts[c] for c in eligible) - 1))
    if n_categorical == 0:
        return SMOTE(k_neighbors=k, sampling_strategy=eligible, random_state=seed)
    return SMOTENC(
        categorical_features=list(range(n_categorical)),
        k_neighbors=k,
        sampling_strategy=eligible,
        random_state=seed,
    )


def run_fold(
    estimator,
    needs_scaling,
    use_smote,
    X_train,
    y_train,
    X_test,
    categorical_cols,
    continuous_cols,
    seed,
):
    """Fit one model on a training fold and predict labels for the untouched test fold."""
    # Labels are re-encoded per fold so models such as XGBoost always see contiguous classes
    encoder = LabelEncoder().fit(y_train)
    y_encoded = encoder.transform(y_train)

    # ColumnTransformer output order: categorical columns first, then continuous ones
    steps = [
        (
            "imputer",
            ColumnTransformer(
                [
                    ("cat", SimpleImputer(strategy="most_frequent"), categorical_cols),
                    ("num", SimpleImputer(strategy="median"), continuous_cols),
                ]
            ),
        )
    ]
    if use_smote:
        sampler = make_sampler(y_encoded, len(categorical_cols), seed)
        if sampler is not None:
            steps.append(("smote", sampler))
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(estimator)))

    pipeline = ImbPipeline(steps).fit(X_train, y_encoded)
    predictions = encoder.inverse_transform(pipeline.predict(X_test))
    return predictions, pipeline, encoder


def cross_validate_model(
    estimator,
    needs_scaling,
    use_smote,
    X,
    y,
    splits,
    categorical_cols,
    continuous_cols,
):
    """Evaluate one model across folds and return metrics and the summed confusion matrix."""
    folds = Parallel(n_jobs=-1)(
        delayed(run_fold)(
            estimator,
            needs_scaling,
            use_smote,
            X.iloc[train],
            y.iloc[train],
            X.iloc[test],
            categorical_cols,
            continuous_cols,
            SEED + i,
        )
        for i, (train, test) in enumerate(splits)
    )

    bal_acc, macro_f1 = [], []
    labels = list(range(len(CLASS_NAMES)))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for (_, test), (pred, _, _) in zip(splits, folds):
        y_true = y.iloc[test]
        bal_acc.append(balanced_accuracy_score(y_true, pred))
        macro_f1.append(f1_score(y_true, pred, average="macro", zero_division=0))
        cm += confusion_matrix(y_true, pred, labels=labels)
    return np.array(bal_acc), np.array(macro_f1), cm, folds


def calculate_feature_importance(
    fold_models: list,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    predictors: list[str],
    model_name: str,
    resampling: str,
) -> pd.DataFrame:
    """Rank predictors by their held-out-fold decrease in balanced accuracy when permuted."""
    importance_values = {column: [] for column in predictors}
    for fold_number, ((_, test_indices), (__, pipeline, encoder)) in enumerate(
        zip(splits, fold_models)
    ):
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        baseline_predictions = encoder.inverse_transform(
            pipeline.predict(X_test).astype(int)
        )
        baseline_score = balanced_accuracy_score(y_test, baseline_predictions)
        rng = np.random.default_rng(SEED + fold_number)

        for column in predictors:
            for _ in range(IMPORTANCE_REPEATS):
                permuted = X_test.copy()
                permuted[column] = rng.permutation(permuted[column].to_numpy())
                permuted_predictions = encoder.inverse_transform(
                    pipeline.predict(permuted).astype(int)
                )
                permuted_score = balanced_accuracy_score(
                    y_test,
                    permuted_predictions,
                )
                importance_values[column].append(baseline_score - permuted_score)

    importance = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Resampling": resampling,
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


def plot_confusion_matrices(matrices: dict, title: str, path: Path) -> None:
    """Save a grid of normalized confusion matrices for each model and resampling mode."""
    models = [m for m in MODELS if m != BASELINE]
    rows = [NO_RESAMPLING, WITH_SMOTE]
    fig, axes = plt.subplots(
        len(rows),
        len(models),
        figsize=(4 * len(models), 8),
        squeeze=False,
    )

    for r, resampling in enumerate(rows):
        for c, model in enumerate(models):
            ax = axes[r][c]
            # Each respondent appears once per repeat, so divide by the repeats for per-respondent counts
            cm = matrices[(model, resampling)] / N_REPEATS
            row_sums = cm.sum(axis=1, keepdims=True)
            pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
            ax.imshow(pct, vmin=0, vmax=1, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:.1f}\n({pct[i, j]:.0%})",
                        ha="center",
                        va="center",
                        color="white" if pct[i, j] > 0.5 else "black",
                        fontsize=9,
                    )
            ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
            ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
            ax.set_title(f"{model}\n[{resampling}]", fontsize=10)
            if c == 0:
                ax.set_ylabel("True class")
            if r == len(rows) - 1:
                ax.set_xlabel("Predicted class")

    fig.suptitle(
        f"{title}\n(avg. respondents per repeat; % = share of the true class)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_target(df: pd.DataFrame, target_col: str, predictors: list, plot_path: Path):
    """Evaluate models and estimate held-out importance for the strongest model variant."""
    print(f"\n{'='*40}")
    print(f"EVALUATING TARGET: {target_col}")
    print(f"{'='*40}")

    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[predictors]
    y = to_classes(df_clean[target_col])

    class_counts = y.value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    print(f"Valid respondents for this target: {len(X)}")
    print(
        "Class distribution: "
        + ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, class_counts))
    )
    if (class_counts > 0).sum() < 2:
        print("SKIPPED: fewer than two classes present, classification is not meaningful.")
        return None, None
    if class_counts[class_counts > 0].min() < N_SPLITS:
        print(
            f"WARNING: a class has fewer than {N_SPLITS} respondents; "
            "folds cannot be fully stratified."
        )

    categorical_cols = [c for c in predictors if c in CATEGORICAL_COLUMNS]
    continuous_cols = [c for c in predictors if c not in CATEGORICAL_COLUMNS]

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )
    splits = list(rskf.split(X, y))

    results, matrices = [], {}
    best_importance_score = float("-inf")
    best_importance_model = None
    best_importance_resampling = None
    best_fold_models = None
    for name, (estimator, needs_scaling) in MODELS.items():
        # SMOTE-NC would make the "most frequent" baseline meaningless, so it only runs without it
        variants = [NO_RESAMPLING] if name == BASELINE else [NO_RESAMPLING, WITH_SMOTE]
        for resampling in variants:
            bal_acc, macro_f1, cm, fold_models = cross_validate_model(
                estimator,
                needs_scaling,
                resampling == WITH_SMOTE,
                X,
                y,
                splits,
                categorical_cols,
                continuous_cols,
            )
            matrices[(name, resampling)] = cm
            if name != BASELINE and bal_acc.mean() > best_importance_score:
                best_importance_score = bal_acc.mean()
                best_importance_model = name
                best_importance_resampling = resampling
                best_fold_models = fold_models
            results.append(
                {
                    "Model": name,
                    "Resampling": resampling,
                    "BalAcc": bal_acc.mean(),
                    "BalAcc_Std": bal_acc.std(),
                    "MacroF1": macro_f1.mean(),
                    "MacroF1_Std": macro_f1.std(),
                }
            )

    res_df = pd.DataFrame(results).sort_values("BalAcc", ascending=False).reset_index(drop=True)
    print("\nRanked best to worst by Balanced Accuracy:")
    print(res_df.round(4).to_string(index=False))

    if (
        best_fold_models is None
        or best_importance_model is None
        or best_importance_resampling is None
    ):
        raise RuntimeError("No non-baseline model was available for feature-importance analysis.")

    importance_df = calculate_feature_importance(
        best_fold_models,
        X,
        y,
        splits,
        predictors,
        best_importance_model,
        best_importance_resampling,
    )
    print(
        f"\nTop predictors for {target_col} "
        f"({best_importance_model}, {best_importance_resampling}; held-out CV permutation):"
    )
    print(importance_df.head(10).round(4).to_string(index=False))

    plot_confusion_matrices(matrices, plot_path.stem.replace("__", " | "), plot_path)
    return res_df, importance_df


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
    return out


def main():
    """Evaluate all datasets and targets, save results, and summarize SMOTE effects."""
    CM_DIR.mkdir(parents=True, exist_ok=True)
    result_rows, importance_rows = [], []

    for formula, predictor_set, path in load_datasets():
        df = pd.read_csv(path)
        predictors = [col for col in df.columns if col not in TARGETS]

        print(f"\n{'#'*64}")
        print(
            f"# TARGET FORMULA: {formula}   |   PREDICTOR SET: {predictor_set} "
            f"({len(predictors)} predictors)"
        )
        print(f"{'#'*64}")

        for target in TARGETS:
            plot_path = CM_DIR / f"{formula}__{predictor_set}__{target}.png"
            res_df, importance_df = evaluate_target(
                df,
                target,
                predictors,
                plot_path,
            )
            if res_df is None:
                continue
            importance_df.insert(0, "Target", target)
            importance_df.insert(0, "Predictors", predictor_set)
            importance_df.insert(0, "Formula", formula)
            importance_rows.extend(importance_df.to_dict("records"))
            for _, row in res_df.iterrows():
                result_rows.append(
                    {
                        "Formula": formula,
                        "Predictors": predictor_set,
                        "Target": target,
                        **row.to_dict(),
                    }
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS_DIR / "framework02_results.csv", index=False)
    feature_importance = pd.DataFrame(importance_rows)
    feature_importance.to_csv(
        RESULTS_DIR / "framework02_feature_importance.csv",
        index=False,
    )

    # Did SMOTE-NC help? Balanced-accuracy / macro-F1 change per model across all runs
    keys = ["Formula", "Predictors", "Target", "Model"]
    plain = results[results["Resampling"] == NO_RESAMPLING].set_index(keys)
    smote = results[results["Resampling"] == WITH_SMOTE].set_index(keys)
    effect = pd.DataFrame(
        {
            "BalAcc_none": plain["BalAcc"],
            "BalAcc_smote": smote["BalAcc"],
            "MacroF1_none": plain["MacroF1"],
            "MacroF1_smote": smote["MacroF1"],
        }
    ).dropna()
    effect["BalAcc_change"] = effect["BalAcc_smote"] - effect["BalAcc_none"]
    effect["MacroF1_change"] = effect["MacroF1_smote"] - effect["MacroF1_none"]
    effect.reset_index().to_csv(
        RESULTS_DIR / "framework02_smote_effect.csv",
        index=False,
    )

    print(f"\n{'#'*64}")
    print(
        "# DID SMOTE-NC HELP? "
        "(change = SMOTE-NC minus no resampling, averaged over all runs)"
    )
    print(f"{'#'*64}")
    per_model = effect.groupby("Model").agg(
        BalAcc_change=("BalAcc_change", "mean"),
        MacroF1_change=("MacroF1_change", "mean"),
        Runs_improved_BalAcc=(
            "BalAcc_change",
            lambda s: f"{(s > 0).sum()}/{len(s)}",
        ),
    )
    print(per_model.round(4).to_string())
    print(
        f"\nOverall: mean balanced-accuracy change "
        f"{effect['BalAcc_change'].mean():+.4f}, "
        f"mean macro-F1 change {effect['MacroF1_change'].mean():+.4f}"
    )

    best = results.loc[
        results.groupby(["Formula", "Predictors", "Target"])["BalAcc"].idxmax()
    ]
    print("\nBest model per dataset and target (by Balanced Accuracy):")
    print(
        best[
            [
                "Formula",
                "Predictors",
                "Target",
                "Model",
                "Resampling",
                "BalAcc",
                "MacroF1",
            ]
        ]
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
                "Resampling",
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
