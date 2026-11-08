"""
PHASE 02 - TARGET NORMALISATION & PREDICTOR COMBINATION
-------------------------------------------------------------------------------------------------
Builds every dataset that Framework 1 evaluates. Two things vary:

  A) HOW THE TARGETS ARE AGGREGATED   -> 3 formulas
  B) HOW THE PREDICTORS ARE PRESENTED -> 2 predictor sets

Every combination is written to its own CSV in datasets/ (3 formulas x 2 predictor sets = 6 files,
named dataset__<formula>__<predictor_set>.csv). 03_framework01.py evaluates every file it finds there.
-------------------------------------------------------------------------------------------------
A) TARGET FORMULAS
-------------------------------------------------------------------------------------------------
Each target combines a primary signal x and a supporting signal y, both first min-max scaled to 0-1
with the survey scale's theoretical range (so the result never depends on the sample):

    target_myopia       x = myopia_level          y = refractive_worsening
    target_cvs          x = cvs_headache_strain   y = cvs_dry_eyes
    target_astigmatism  x = astigmatism_symptoms  y = myopia_level

    linear    w*x + (1-w)*y              Weighted mean. Interpretable; a high x can be offset by a low y.
    power     sqrt(w*x^2 + (1-w)*y^2)    Weighted quadratic mean (RMS-style). One severe signal counts for
                                         more than two mild ones.
    noisy_or  1 - (1-x)*(1-y)            Independent evidence: any strong signal pushes the score up.
                                         Ignores the weight and gives the highest scores of the three.

If x or y is missing, the target is missing and that respondent is dropped for that target only
(e.g. 'Unsure' myopia answers also remove respondents from the astigmatism target).

-------------------------------------------------------------------------------------------------
B) PREDICTOR SETS
-------------------------------------------------------------------------------------------------
    raw       the 14 predictors exactly as encoded in 01_preprocessing.py.
    combined  predictors that plausibly act together are merged into composite "strain" indices
              (see PREDICTOR_GROUPS); predictors that belong to no group are passed through unchanged.

Composite indices only use fixed, theoretical scale ranges and fixed weights, never statistics learned
from the data, so building them cannot leak information between training and test folds.
-------------------------------------------------------------------------------------------------
Input:  preprocessing_output/survey_encoded_model_candidate.csv (from 01_preprocessing.py)
Output: datasets/dataset__<formula>__<predictor_set>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


SCALE = {
    "myopia_level": (1, 5),
    "refractive_worsening": (1, 4),
    "cvs_headache_strain": (1, 5),
    "cvs_dry_eyes": (1, 5),
    "astigmatism_symptoms": (1, 5),
}

TARGET_SPECS = {
    "target_myopia": ("myopia_level", "refractive_worsening", 0.7),
    "target_cvs": ("cvs_headache_strain", "cvs_dry_eyes", 0.5),
    "target_astigmatism": ("astigmatism_symptoms", "myopia_level", 0.7),
}


def combine_linear(x, y, w):
    """Combine two scaled target signals using their weighted arithmetic mean."""
    return w * x + (1 - w) * y


def combine_power(x, y, w):
    """Combine two scaled target signals using their weighted quadratic mean."""
    return np.sqrt(w * x**2 + (1 - w) * y**2)


def combine_noisy_or(x, y, w):
    """Combine two scaled target signals using the noisy-OR formula."""
    return 1 - (1 - x) * (1 - y)


TARGET_FORMULAS = {
    "linear": combine_linear,
    "power": combine_power,
    "noisy_or": combine_noisy_or,
}


PREDICTOR_SCALE = {
    "daily_screen_time": (1, 8),
    "continuous_usage": (1, 5),
    "intensity": (1, 5),
    "lighting": (1, 5),
    "multi_device": (1, 5),
    "phone_distance": (1, 5),
    "monitor_distance": (1, 5),
    "before_bed_usage": (1, 5),
    "blue_light_filter": (1, 5),
    "outdoor_activity": (1, 5),
    "genetics": (1, 5),
}


PREDICTOR_GROUPS = {
    "screen_load": [
        ("daily_screen_time", False, 1.0),
        ("continuous_usage", False, 1.0),
        ("intensity", False, 1.0),
        ("multi_device", False, 1.0),
    ],
    "viewing_strain": [
        ("phone_distance", True, 1.0),
        ("monitor_distance", True, 1.0),
        ("lighting", False, 1.0),
    ],
    "night_exposure": [
        ("before_bed_usage", False, 1.0),
        ("blue_light_filter", True, 1.0),
    ],
    "family_risk": [("genetics", False, 1.0)],
    "protective_outdoor": [("outdoor_activity", False, 1.0)],
}

TARGET_INPUT_COLUMNS = [
    "myopia_level",
    "refractive_worsening",
    "cvs_headache_strain",
    "cvs_dry_eyes",
    "astigmatism_symptoms",
]


def min_max_scale(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """Normalize a series to 0-1 using the scale's theoretical minimum and maximum."""
    return (series.astype(float) - min_val) / (max_val - min_val)


def build_targets(df: pd.DataFrame, combine) -> pd.DataFrame:
    """Build the three normalized target columns using the supplied combination formula."""
    out = {}
    for target, (x_col, y_col, w) in TARGET_SPECS.items():
        x = min_max_scale(df[x_col], *SCALE[x_col])
        y = min_max_scale(df[y_col], *SCALE[y_col])
        out[target] = combine(x, y, w)
    return pd.DataFrame(out, index=df.index)


def build_group_index(df: pd.DataFrame, members: list) -> pd.Series:
    """Build a weighted mean index from scaled members, ignoring missing values."""
    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for col, invert, weight in members:
        scaled = min_max_scale(df[col], *PREDICTOR_SCALE[col])
        if invert:
            scaled = 1 - scaled
        present = scaled.notna()
        numerator += (weight * scaled).where(present, 0.0)
        denominator += weight * present
    return numerator / denominator.where(denominator > 0)


def build_predictor_sets(df: pd.DataFrame, predictor_columns: list) -> dict:
    """Return raw predictors and a set where related predictors are combined into indices."""
    grouped = [col for members in PREDICTOR_GROUPS.values() for col, _, _ in members]
    duplicates = {c for c in grouped if grouped.count(c) > 1}
    if duplicates:
        raise ValueError(f"Predictors used in more than one group: {sorted(duplicates)}")
    unknown = [c for c in grouped if c not in predictor_columns or c not in PREDICTOR_SCALE]
    if unknown:
        raise ValueError(
            f"Group members must be predictors listed in PREDICTOR_SCALE: {unknown}"
        )

    combined = pd.DataFrame(index=df.index)
    for name, members in PREDICTOR_GROUPS.items():
        combined[name] = build_group_index(df, members)
    for col in predictor_columns:
        if col not in grouped:
            combined[col] = df[col]

    return {"raw": df[predictor_columns].copy(), "combined": combined}


def main():
    """Parse command-line options and write all target and predictor dataset variants."""
    parser = argparse.ArgumentParser(
        description="Construct Framework 1 datasets."
    )

    parser.add_argument(
        "--input",
        default=str(HERE / "preprocessing_output" / "survey_encoded_model_candidate.csv"),
        help="Path to the model candidate CSV from preprocessing.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(HERE / "datasets"),
        help="Directory to save the datasets.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run 01_preprocessing.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob("dataset__*.csv"):
        old.unlink()

    print(f"Loading {input_path.name}...")
    df = pd.read_csv(input_path)
    predictor_columns = [c for c in df.columns if c not in TARGET_INPUT_COLUMNS]
    predictor_sets = build_predictor_sets(df, predictor_columns)

    for set_name, predictors in predictor_sets.items():
        print(
            f"\nPredictor set '{set_name}' ({predictors.shape[1]} columns): "
            f"{list(predictors.columns)}"
        )

    for formula_name, combine in TARGET_FORMULAS.items():
        targets = build_targets(df, combine)

        print(f"\n--- Target formula '{formula_name}' (0.0 to 1.0 scale) ---")
        print(targets.describe().round(3))

        for set_name, predictors in predictor_sets.items():
            out_file = output_dir / f"dataset__{formula_name}__{set_name}.csv"
            pd.concat([predictors, targets], axis=1).to_csv(out_file, index=False)
            print(
                f"Saved: {out_file.name}  (rows: {len(targets)}, "
                f"missing per target: {targets.isna().sum().to_dict()})"
            )


if __name__ == "__main__":
    main()
