"""
PART 01E: Borderline SMOTE for Machine Learning Pipeline

Borderline SMOTE variant that focuses on borderline minority samples.
Instead of oversampling all minority samples, it identifies samples
near the decision boundary (borderline samples) and generates synthetic
samples near these critical regions. This approach is ideal for small
datasets where focusing on difficult-to-classify samples is more
valuable than blindly oversampling the entire minority class.

Benefits:
- Focuses on difficult-to-classify borderline samples
- Better for small datasets (like our 81-sample survey)
- Generates synthetic samples near decision boundaries
- No clustering required (unlike KMeans SMOTE)
- More efficient than vanilla SMOTE for small data

The script follows the same logic as other SMOTE variants:
1. Load and normalize inputs
2. Remove singleton values
3. Create target scores using WEIGHTED AVERAGE formula
4. Apply Borderline SMOTE to balance classes
5. Output three CSV files for each target variable
"""

from numpy.linalg import norm
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from imblearn.over_sampling import BorderlineSMOTE
import os

def normalize_column(col):
    min_val = col.min()
    max_val = col.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(col))
    return (col - min_val) / (max_val - min_val)

def get_valid_normalized_values(col):
    min_val = col.min()
    max_val = col.max()
    
    if max_val == min_val:
        return np.array([0.5])
    
    possible_original_values = np.arange(min_val, max_val + 1)
    
    valid_normalized = (possible_original_values - min_val) / (max_val - min_val)
    
    return valid_normalized

def round_to_valid_values(synthetic_data, original_df):
    rounded_data = synthetic_data.copy()

    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx'
    ]
    
    for col in input_cols:
        if col in synthetic_data.columns:
            valid_values = get_valid_normalized_values(original_df[col])
            
            for idx in rounded_data.index:
                synthetic_val = rounded_data.loc[idx, col]
                nearest_idx = np.argmin(np.abs(valid_values - synthetic_val))
                rounded_data.loc[idx, col] = valid_values[nearest_idx]
    
    return rounded_data

def load_and_normalize_inputs(csv_file):
    df = pd.read_csv(csv_file)
    
    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx', 'myopia_level', 'refractive_worsening', 'cvs_headache_strain',
        'cvs_dry_eyes', 'astigmatism_symptoms'
    ]
    
    normalized_df = pd.DataFrame()

    for col in input_cols:
        normalized_df[col] = normalize_column(df[col])
        

    for col in input_cols:
        value_counts = normalized_df[col].value_counts()
        singleton_values = value_counts[value_counts == 1].index
        normalized_df = normalized_df[~normalized_df[col].isin(singleton_values)]
        
        if len(singleton_values) > 0:
            print(f"Removed {len(singleton_values)} singleton values from '{col}'")
        
  
    # WEIGHTED AVERAGE APPROACH - OPTIMIZED WEIGHTS
    # Myopia: 25% myopia_level + 75% refractive_worsening
    # CVS: 50% headaches + 50% dry eyes
    # Astigmatism: 50% symptoms + 50% myopia
    
    normalized_df["myopia_score"] = (
        (normalized_df["myopia_level"]) * 0.25 + (normalized_df["refractive_worsening"]) * 0.75
    )

    normalized_df["cvs_score"] = (
        (normalized_df["cvs_headache_strain"]) * 0.50 + (normalized_df["cvs_dry_eyes"]) * 0.50
    )

    normalized_df["astigmatism_score"] = (
        (normalized_df["astigmatism_symptoms"]) * 0.50 + (normalized_df["myopia_level"]) * 0.50
    )
    
    return normalized_df

def apply_borderline_smote_for_ml(df, primary_target):
    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx', 'myopia_level', 'refractive_worsening', 'cvs_headache_strain',
        'cvs_dry_eyes', 'astigmatism_symptoms'
    ]

    X = df[input_cols].copy().reset_index(drop=True)
    y_continuous = df[primary_target].copy().reset_index(drop=True)
    
    # Convert continuous scores to ordinal integer classes (0 to 4)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [0, 1, 2, 3, 4]
    y = pd.cut(y_continuous, bins=bins, labels=labels, include_lowest=True)
    y = pd.factorize(y)[0]

    print(f"  Class distribution before filtering (using {primary_target}):")
    print(f"  {pd.Series(y).value_counts().sort_index().to_dict()}")
    print(f"  Total samples: {len(df)}")

    # Remove classes with fewer than 2 samples (SMOTE needs at least 2 per class)
    y_series = pd.Series(y, index=X.index)
    class_counts = y_series.value_counts()
    classes_to_keep = class_counts[class_counts >= 2].index
    mask = y_series.isin(classes_to_keep)
    X = X[mask].reset_index(drop=True)
    y = y_series[mask].values
    
    print(f"  Removed classes with <2 samples")
    print(f"  Class distribution after filtering (using {primary_target}):")
    print(f"  {pd.Series(y).value_counts().sort_index().to_dict()}")
    print(f"  Total samples: {len(X)}")

    min_class_size = pd.Series(y).value_counts().min()
    k_neighbors = max(1, min_class_size - 1)
    print(f"  Using k_neighbors={k_neighbors} for Borderline SMOTE")

    # Borderline SMOTE: focuses on borderline (difficult-to-classify) minority samples
    # m_neighbors: number of neighbors to decide if sample is borderline (1-6, we use 5)
    borderline_smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors, m_neighbors=5, kind='borderline-1')
    X_resampled, y_resampled = borderline_smote.fit_resample(X, y)

    print(f"  Class distribution after Borderline SMOTE:")
    print(f"  {pd.Series(y_resampled).value_counts().sort_index().to_dict()}")
    print(f"  Total samples after Borderline SMOTE: {len(X_resampled)}")

    print(f"  Rounding synthetic values to valid ordinal ranges...")
    X_resampled = round_to_valid_values(X_resampled, df)
    print(f"  Rounding complete!")

    for col in input_cols:
        X_resampled[col] = np.clip(X_resampled[col], 0.0, 1.0)

    balanced_df = pd.DataFrame(X_resampled, columns=input_cols)
    balanced_df[primary_target] = y_resampled

    return balanced_df

def main():
    csv_file = "../OUTPUT.csv"
    
    print("Loading and normalizing survey responses...")
    df = load_and_normalize_inputs(csv_file)
    
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    
    print("\nApplying Borderline SMOTE with Weighted Average targets...")
    myopia_balanced_df = apply_borderline_smote_for_ml(df, primary_target="myopia_score")
    
    myopia_output = output_dir / "training_data_smote_myopia.csv"
    myopia_balanced_df.to_csv(myopia_output, index=False)
    print(f"  Saved myopia training data to {myopia_output}")
    
    cvs_balanced_df = apply_borderline_smote_for_ml(df, primary_target="cvs_score")
    cvs_output = output_dir / "training_data_smote_computervs.csv"
    cvs_balanced_df.to_csv(cvs_output, index=False)
    print(f"  Saved CVS training data to {cvs_output}")
    
    astigmatism_balanced_df = apply_borderline_smote_for_ml(df, primary_target="astigmatism_score")
    astigmatism_output = output_dir / "training_data_smote_astigmatism.csv"
    astigmatism_balanced_df.to_csv(astigmatism_output, index=False)
    print(f"  Saved astigmatism training data to {astigmatism_output}")
    
    print("\nAll Borderline SMOTE training data generated successfully!")

if __name__ == "__main__":
    main()
