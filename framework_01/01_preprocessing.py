"""
PHASE 01 - PREPROCESSING THE ORIGINAL DATA & ENCODING CATEGORIES
-------------------------------------------------------------------------------------------------
Loads the survey CSV, validates and renames its columns, encodes responses, and writes audit data and
distribution reports to the selected output directory.

The following script does data preprocessing upon the original survey response.
The columns used for preprocessing and formating are pre-defined and will only function properly on the
exact same CSV format given when doing the initial research.

If a new dataset is wanted to be used using this same pre-processing, the following two steps are required:
-   The COLUMN_RENAME constant dictionary needs to be adapted so that it contains the exact same column names as
    the new dataset has.
-   Similarly, the new dataset needs to be mapped to the same names of the COLUMN_RENAME values. THE VALUES SHOULD NOT
    BE CHANGED AS THEIR NAMES ARE USED PRE-DEFINED IN THE FURTHER STEPS.

This pipeline follows these steps in this order:
-   Initially, upon specifying the command line arguments for the CSV path and the output directory (or by simply running
    the default values) the script loads the original CSV file of the survey and creates the output directory for the preprocesing.
-   Upon loading the CSV file, the script renames the columns of the now created DataFrame into smaller version and checks if the
    CSV file itself is properly formatted.
-   Afterwards, the encoding process drops Myopia related target rows which have a null value in order to prevent data misconception and 
    innacurate learning. It then converts every verbally made answer within the CSV itself and turns it into an ordinal value. It
    repeats this for every value within the CSV sheet.

By default using the initial dataset, 99 responses will be formatted. Out of those 99, 13 will be dropped for Myopia and Astigmatism as
the value "myopia_score" was left empty by 13 users.

It exports images of the variables distributions, a Spearman coefficient matrix as well as generate the CSV files of the preprocsssing data,
which includes survey_encoded_model_candidate.csv, the main file used later on in the analysis.

Execution: 01_preprocessing.py --csv_file /path/to/csv/file --output /path/to/output/dir
-------------------------------------------------------------------------------------------------
"""
import argparse
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default value setting for the paths of the CSV file and the output directory
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "INPUT_ORG.csv"
DEFAULT_OUTPUT = HERE / "preprocessing_output"

COLUMN_RENAME = {
    "Timestamp": "timestamp",
    "What is Your age?": "age",
    "What is Your gender?": "gender",
    "On a typical day, how many hours do you spend looking at digital screens in total? (phones, tablets, computers, TV)? Select the option that matches closest.": "daily_screen_time",
    "How long do you typically use a screen continuously before taking a break? (without looking away for at least 20 seconds)": "continuous_usage",
    "How would you describe the typical intensity of your screen-based tasks?": "intensity",
    "How would you describe the lighting conditions in which you typically use your devices?": "lighting",
    "How often do you use two or more screens simultaneously? (e.g. laptop + phone at the same time)": "multi_device",
    "How far do you typically hold your phone from your eyes when using it for messaging, social media, scrolling and other tasks?": "phone_distance",
    "How far is your desktop monitor, laptop screen or other main screen typically placed from your eyes when using it?": "monitor_distance",
    'How consistently do you use Blue Light Filters, "Night Shift," or "Night Mode" on your devices?': "blue_light_filter",
    "How often do you use any digital screen within 1 hour of going to sleep?": "before_bed_usage",
    "Which category best describes your daily screen usage in terms of your professional or academic environment?": "profession",
    "On average, how much time do you spend outdoors in natural light each day?": "outdoor_activity",
    "To your knowledge, do your biological parents have vision prescriptions / errors? (Nearsightness / Astigmatism / Blurred Vision)?": "genetics",
    "At what age did You receive your first prescription? Enter 0 if you never received a prescription or are really not sure at what age you received your first.": "age_first_rx",
    'What is the measure of your current nearsightedness prescription? Please select the measurement closest to the prescription you remember you have. Select "Unsure" if you are really unsure.': "myopia_level",
    "Over the last 2 years, how has your ability to see objects at a distance changed?": "refractive_worsening",
    "How often do you experience physical symptoms such as headaches, eye-strain and similar?": "cvs_headache_strain",
    "How often do you experience dry eyes and focus issues after prolonged digital usage?": "cvs_dry_eyes",
    'When looking at text, bright lights or specifically lights at night, how often do you see "ghosting," shadows, or double-edges?': "astigmatism_symptoms",
}

EXPECTED_COLUMNS = list(COLUMN_RENAME.values())

PREDICTOR_COLUMNS = [
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
    "age_first_rx",
    "has_prescription",
]

TARGET_INPUT_COLUMNS = [
    "myopia_level",
    "refractive_worsening",
    "cvs_headache_strain",
    "cvs_dry_eyes",
    "astigmatism_symptoms",
]

DEMOGRAPHIC_COLUMNS = ["age", "gender"]

# ============================================================ #
# ============= DATA ENCODING AND CLEANING START ============= #
# ============================================================ #

def clean_text(value: object) -> str:
    """Normalize a response to lowercase, trimmed text for consistent matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text

def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether the text contains at least one of the supplied terms."""
    return any(term in text for term in terms)


def encode_ordinal(column: str, value: object) -> Optional[int]:
    """Map a survey response to its column-specific ordinal rank, or None if unmapped."""
    t = clean_text(value)
    if not t:
        return None

    
    if column == "daily_screen_time":
        if contains_any(t, ("less than 1", "<1", "under 1")): return 1
        if "1 to 2" in t or "1-2" in t: return 2
        if "2 to 3" in t or "2-3" in t: return 3
        if "3 to 4" in t or "3-4" in t: return 4
        if "4 to 5" in t or "4-5" in t: return 5
        if "5 to 6" in t or "5-6" in t: return 6
        if "6 to 7" in t or "6-7" in t: return 7
        if contains_any(t, ("more than 7", "7+")): return 8

    
    elif column == "continuous_usage":
        if "10" in t and "20" in t: return 1
        if "20" in t and "30" in t: return 2
        if "30" in t and "40" in t: return 3
        if "40" in t and "60" in t: return 4
        if contains_any(t, ("more than 60", "over 60", ">60")): return 5

    
    elif column == "intensity":
        if contains_any(t, ("low focus", "low intensity")): return 1
        if "below average" in t: return 2
        if t.startswith("average") or "moderate intensity" in t: return 3
        if "above average" in t: return 4
        if contains_any(t, ("very high", "high intensity")): return 5

    
    elif column == "lighting":
        if contains_any(t, ("almost exclusively bright", "exclusively bright")): return 1
        if "mostly bright" in t: return 2
        if contains_any(t, ("even mix", "balanced mix", "mix of bright and dim")): return 3
        if "mostly dim" in t: return 4
        if contains_any(t, ("almost exclusively dim", "exclusively dim", "exclusively in dark", "dark/dim")): return 5

    
    elif column in {
        "multi_device", "blue_light_filter", "before_bed_usage",
        "cvs_headache_strain", "cvs_dry_eyes", "astigmatism_symptoms"
    }:
        
        if contains_any(t, ("never", "not at all", "clear edges")): return 1
        if "rarely" in t: return 2
        if contains_any(t, ("occasionally", "sometimes")): return 3
        if contains_any(t, ("frequently", "often")): return 4
        if contains_any(t, ("almost always", "constantly", "almost daily", "always", "every night")): return 5

    
    elif column == "phone_distance":
        if contains_any(t, ("less than 15", "<15")): return 1
        if "15" in t and "25" in t: return 2
        if "25" in t and "35" in t: return 3
        if "35" in t and "50" in t: return 4
        if contains_any(t, ("more than 50", ">50")): return 5

    elif column == "monitor_distance":
        if contains_any(t, ("less than 20", "<20")): return 1
        if "20" in t and "35" in t: return 2
        if "35" in t and "50" in t: return 3
        if ("50" in t and "1m" in t) or ("50 cm" in t and "1 m" in t): return 4
        if contains_any(t, ("more than 1m", "more than 1 m", ">1m", ">1 m")): return 5

    
    elif column == "profession":
        if contains_any(t, ("no digital use", "manual", "outdoor")): return 1
        if contains_any(t, ("low", "retail")): return 2
        if contains_any(t, ("moderate", "office")): return 3
        if contains_any(t, ("high", "student", "research")): return 4
        if contains_any(t, ("extreme", "software engineering", "data analysis", "pro gaming")): return 5

    
    elif column == "outdoor_activity":
        if contains_any(t, ("less than 30", "<30")): return 1
        if ("30" in t and "1 hour" in t) or ("30 min" in t and "1h" in t): return 2
        if "1" in t and "2" in t and "hour" in t: return 3
        if "2" in t and "3" in t and "hour" in t: return 4
        if contains_any(t, ("more than 3", ">3")): return 5

    
    elif column == "genetics":
        if contains_any(t, ("no history", "neither parent", "none")): return 1
        if "one" in t and "mild" in t: return 2
        if "one" in t and "strong" in t: return 3
        if "both" in t and contains_any(t, ("mild", "moderate")): return 4
        if "both" in t and "strong" in t: return 5

    
    
    elif column == "myopia_level":
        if contains_any(t, ("unsure", "not sure", "don't know", "do not know")): return None
        if contains_any(t, ("0.0 to -0.5", "0 to -0.5", "no prescription", "very mild")): return 1
        if "-0.5 to -1.5" in t or "-0.5 - -1.5" in t: return 2
        if "-1.5 to -3.0" in t or "-1.5 - -3.0" in t: return 3
        if "-3.0 to -4.5" in t or "-3.0 - -4.5" in t: return 4
        if contains_any(t, ("-4.5 and below", "-4.5 or below", "more than -4.5")): return 5

    
    elif column == "refractive_worsening":
        if contains_any(t, ("stable", "no change", "same")): return 1
        if "slightly" in t: return 2
        if "moderately" in t or "moderate" in t: return 3
        if "significantly" in t or "significant" in t: return 4

    return None

# ========================================================== #
# ============= DATA ENCODING AND CLEANING END ============= #
# ========================================================== #

# ============================================================== #
# ============= SCRIPT LODAING AND DIRECTORY START ============= #
# ============================================================== #

def make_output_dirs(output_dir: Path) -> None:
    """Create the output directory and its plot subdirectories if needed."""
    dirs = {
        "root": output_dir,
        "plots": output_dir / "plots",
        "distributions": output_dir / "plots" / "distributions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

def load_and_rename(csv_path: Path) -> pd.DataFrame:
    """Read the survey CSV, validate its expected headers, and rename its columns."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing_headers = []

    for h in COLUMN_RENAME:
        if h not in df.columns:
            missing_headers.append(h)
            
    if missing_headers:
        print("\nWARNING: Some expected original headers were not found:")
        for h in missing_headers:
            print(f"  - {h}")
        print("\nHeaders actually found:")
        for h in df.columns:
            print(f"  - {h}")
        raise ValueError(
            "CSV headers do not exactly match the expected Google Forms export. "
            "Update COLUMN_RENAME using the headers printed above."
        )

    df = df.rename(columns=COLUMN_RENAME)
    return df


def transform_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode survey fields and return the transformed data plus unmapped responses."""
    encoded = pd.DataFrame(index=raw.index)

    encoded["timestamp"] = raw["timestamp"] if "timestamp" in raw else np.nan
    
    for col in ["age", "age_first_rx"]:
        encoded[col] = pd.to_numeric(raw[col], errors="coerce")
    
    encoded["gender"] = raw["gender"].astype("string").str.strip()

    encoded["has_prescription"] = raw["age_first_rx"].apply(
        lambda x: 0 if pd.isna(x) or x == 0 else 1
    )

    unmapped_rows = []
    ordinal_columns = [
        "daily_screen_time", "continuous_usage", "intensity", "lighting",
        "multi_device", "phone_distance", "monitor_distance",
        "blue_light_filter", "before_bed_usage", "profession",
        "outdoor_activity", "genetics", *TARGET_INPUT_COLUMNS
    ]
    
    for col in ordinal_columns:
        values = []
        for idx, original_value in raw[col].items():
            code = encode_ordinal(col, original_value)
            values.append(code)
            if pd.notna(original_value) and str(original_value).strip() and code is None:
                unmapped_rows.append({
                    "row_number_in_csv": int(idx) + 2,
                    "column": col,
                    "original_response": str(original_value),
                })
        encoded[col] = pd.array(values, dtype="Int64")

    unmapped = pd.DataFrame(
        unmapped_rows,
        columns=["row_number_in_csv", "column", "original_response"]
    )
    return encoded, unmapped

def build_distribution_reports(df: pd.DataFrame, out: Path) -> None:
    """Write missingness, summary, category, plot, and Spearman-correlation reports."""
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 2) for c in df.columns
        ],
    })
    missing.to_csv(out / "missing_values.csv", index=False)

    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        df[numeric_cols].describe().T.to_csv(out / "numerical_statistics.csv")

    
    frequency_rows = []
    for col in df.columns:
        if col == "timestamp":
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        counts = series.value_counts(dropna=False).sort_index()
        total_non_missing = len(series)
        for category, count in counts.items():
            frequency_rows.append({
                "column": col,
                "category_or_value": category,
                "count": int(count),
                "percent_of_non_missing": round(count / total_non_missing * 100, 2),
                "total_non_missing": total_non_missing,
            })

    pd.DataFrame(frequency_rows).to_csv(
        out / "categorical_distributions.csv", index=False
    )

    
    plots_dir = out / "plots" / "distributions"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        if col == "timestamp":
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        plt.figure(figsize=(9, 5))
        if pd.api.types.is_numeric_dtype(series):
            
            plt.hist(series.astype(float), bins="auto", edgecolor="black")
            plt.ylabel("Respondent count")
        else:
            counts = series.astype(str).value_counts()
            plt.bar(counts.index, counts.values)
            plt.ylabel("Respondent count")
            plt.xticks(rotation=35, ha="right")

        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", col)
        plt.savefig(plots_dir / f"{safe_name}.png", dpi=160)
        plt.close()

    
    
    corr_cols = [
        c for c in df.columns
        if c not in {"timestamp", "gender", "has_prescription", "age"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(method="spearman")
        corr.to_csv(out / "spearman_correlations.csv")

        plt.figure(figsize=(12, 10))
        image = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(image, label="Spearman correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Exploratory Spearman correlation matrix")
        plt.tight_layout()
        plt.savefig(out / "plots" / "spearman_correlation.png", dpi=160)
        plt.close()


# ============================================================ #
# ============= SCRIPT LODAING AND DIRECTORY END ============= #
# ============================================================ #

def main() -> None:
    """Run the preprocessing pipeline using command-line input and output paths."""
    parser = argparse.ArgumentParser(
        description="Preprocess and audit the ocular-health survey CSV.",
        epilog="Example usage: 01_preprocessing.py --csv_path path/to/input.csv --output path/to/output/dir"
    )
    parser.add_argument(
        "--csv_path",
        default=str(DEFAULT_INPUT),
        help="Path to the original Google Forms CSV (default: ../INPUT_ORG.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory (default: framework_01/preprocessing_output).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    make_output_dirs(output_dir)

    raw = load_and_rename(csv_path)
    print(f"Loaded {len(raw)} respondents and {len(raw.columns)} columns.")

    encoded, unmapped = transform_dataset(raw)

    encoded.to_csv(output_dir / "survey_encoded_audit.csv", index=False)
    model_candidate = encoded[PREDICTOR_COLUMNS + TARGET_INPUT_COLUMNS].copy() 
    model_candidate.to_csv(output_dir / "survey_encoded_model_candidate.csv", index=False)

    unmapped.to_csv(output_dir / "unmapped_categories.csv", index=False)

    build_distribution_reports(encoded, output_dir)

    print("\nPreprocessing and inspection complete.")
    print(f"Output directory: {output_dir}")
    print(f"Respondents: {len(encoded)}")
    print(f"Unmapped non-empty responses: {len(unmapped)}")
    print(f"\nGenerated distribution report figures and Spearman coefficient matrix figure in {output_dir}/plots")

    if not unmapped.empty:
        print(
            "\nIMPORTANT: Review unmapped_categories.csv. "
            "Unmapped answers remain missing; these are the answers which the encoder could not convert " \
            "to meaningfull results. Proceed with the analysis with caution, or resolve the unmapped entries directly." \
            "The rows which were not mapped remain unlisted in the final encoded dataset."
        )


if __name__ == "__main__":
    main()
