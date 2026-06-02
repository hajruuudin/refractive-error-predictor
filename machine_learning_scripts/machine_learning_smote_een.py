"""
==== SMOTE with Edited Nearest Neighbour Clensing ====

This variation of SMOTE does the same exact process as regular SMOTE (or SMOTE NC), with one additional feature.
After performing synthetic generation, SMOTE EEN runs an Edited NEarest Neighbour clensing process, where
it checks each new synthetic sample against its surrounding samples. In the case where it finds
that the majority of the samples found within the neighbours are different from the newly generated
sample, it determines that the new sample is an outlier and only causes noise instead of helping 
the overall final predictions.

SMOTE-EEN turned out to be the most effective approach used within this survey paper. However, note that
the small sample size of the original survey and the nature of this approach means that some of the 
predictive accuracy could be mistaken for a combination of luck and the relatively small size of the dataset.
"""


from numpy.linalg import norm
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import os
from target_equations import TargetEquations

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
        
    # WEIGHTED AVERAGE APPROACH
    (
    normalized_df["myopia_score"],
    normalized_df["computervs_score"],
    normalized_df["astigmatism_score"]
    ) = TargetEquations.calculate_weighted_average(self=TargetEquations, normalized_df=normalized_df)
    
    # # POWER MEAN APPROACH
    # (
    # normalized_df["myopia_score"],
    # normalized_df["computervs_score"],
    # normalized_df["astigmatism_score"]
    # ) = TargetEquations.calculate_power_mean(normalized_df)
   
    
    # # GEOMETRIC MEAN APPROACH
    # (
    # normalized_df["myopia_score"],
    # normalized_df["computervs_score"],
    # normalized_df["astigmatism_score"]
    # ) = TargetEquations.calculate_geometric_mean(normalized_df)
   
    # # NORMALISED MEAN APPROACT
    # (
    # normalized_df["myopia_score"],
    # normalized_df["computervs_score"],
    # normalized_df["astigmatism_score"]
    # ) = TargetEquations.calculate_normalized_sum(normalized_df)

    # # MAX PENALTY APPROACH
    # (
    # normalized_df["myopia_score"],
    # normalized_df["computervs_score"],
    # normalized_df["astigmatism_score"]
    # ) = TargetEquations.calculate_max_penalty(normalized_df)
   
    # # HARMONIC MEAN APPROACH
    # (
    # normalized_df["myopia_score"],
    # normalized_df["computervs_score"],
    # normalized_df["astigmatism_score"]
    # ) = TargetEquations.calculate_harmonic_mean(normalized_df)
    
    return normalized_df

def apply_smote_een_for_ml(df, primary_target):
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
    print(f"  Using k_neighbors={k_neighbors} for SMOTE-EEN")

    # Configure SMOTE with appropriate k_neighbors for this dataset
    smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy='auto', random_state=42)
    smote_een = SMOTEENN(smote=smote, sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote_een.fit_resample(X, y)

    print(f"  Class distribution after SMOTE-EEN:")
    print(f"  {pd.Series(y_resampled).value_counts().sort_index().to_dict()}")
    print(f"  Total samples after SMOTE-EEN: {len(X_resampled)}")

    print(f"  Rounding synthetic values to valid ordinal ranges...")
    X_resampled = round_to_valid_values(X_resampled, df)
    print(f"  Rounding complete!")

    for col in input_cols:
        X_resampled[col] = np.clip(X_resampled[col], 0.0, 1.0)

    balanced_df = pd.DataFrame(X_resampled, columns=input_cols)
    balanced_df[primary_target] = y_resampled

    return balanced_df

def main():
    parser = argparse.ArgumentParser(description='Encode survey CSV using SMOTE-EEN.')
    parser.add_argument('--input_path', default='../OUTPUT.csv')
    parser.add_argument('--output_prefix', default='training_data_smote_')
    
    args = parser.parse_args()
 
    input_path  = Path(args.input_path)
    output_prefix = Path(args.output_prefix)

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not correlate to an adequate CSV file")
        return
    
    print("=" * 70)
    print("MACHINE LEARNING - SMOTE-EEN DATA AUGMENTATION")
    print("The prefix is:")
    print(output_prefix)
    print("=" * 70)
    
    print(f"\nLoading and normalizing survey responses...")
    df = load_and_normalize_inputs(input_path)
    print(f"Loaded {len(df)} survey responses with 15 normalized inputs")
    
    print(f"\nApplying SMOTE-EEN with Weighted Average targets...")
    myopia_balanced_df = apply_smote_een_for_ml(df, primary_target="myopia_score")
    computervs_balanced_df = apply_smote_een_for_ml(df, primary_target="cvs_score")
    astigmatism_balanced_df = apply_smote_een_for_ml(df, primary_target="astigmatism_score")
    
    print(f"\nSaving balanced dataset...")
    myopia_balanced_df.to_csv(str(output_prefix) + "myopia.csv", index=False)
    computervs_balanced_df.to_csv(str(output_prefix) + "computervs.csv", index=False)
    astigmatism_balanced_df.to_csv(str(output_prefix) + "astigmatism.csv", index=False)
    
    print(f"Saved individual datasets into respective CSV files")
    
    print("\n" + "=" * 70)
    print("SMOTE-EEN processing complete! Dataset ready for individual ML model training.")
    print("=" * 70)

if __name__ == "__main__":
    main()
