"""
PART 01: SMOTE for Machine Learning Pipeline

For the ML model, SMOTE is required due to the expected small size in the dataset, as well as to fix underlying imbalance problems
for the minority classes. The following script evens out the dataset and prepares three individual CSV files 
which are later used as data inputs for the ML model training, each of which are extended
based on one of the three expected target variables (MYOPIA, CSV OR ASTIGMATISM). The reason why we create three individual
dataset instead of using K-Nearest-Neighbour for the missing target data is due to the small size of the input survey. This is
to avoid noisy data and inaccurate/non-sensible values for the remaining target variables.

The following script works in the steps in order:
    1. Load the original CSV file containing the survey samples (or the fake responses just for testing purposes)
    2. The inputs are normalised, so that the values are not categorical but rather all ordinal from 0.0 to 1.0
    3. The target columns are aggregated so that we have 3 targets instead of 5
    4. Then, the SMOTE process is applied three consecutive times, all separate
        First for Myopia, then for CVS and then or Astigmatism
    5. SMOTE is set up so that the K-neighbours variables is by default 3, however, can
        drop to 2 if the minority class does not have enough instances (can cause detriments to the effect tho)
    6. The process is repeated 3 times, once for each target
    7. The result is three CSV files. Each of them is used later for training of ML models.

Some things to note:
    - The functions that aggregate the 5 target variables into three are as follows:
        - For Myopia: It is the average of the current nearsightedness score and the worsening of the individuals eyesight in the last 2 years
        - For Computer Vision Syndrome: It is the average of the frequency of experiencing headaches and digital strain
        - For astigmatism: It is the astigmatism symptoms with an addition of 10% of the users nearsightedness score
      Each of the formulas can be changed and it changes the overall target
    - In this current setup, there is no usage of K-Nearest-Neighbour. The reason for this is the lack
      of size in the input survey. A usage of KNN would mean narrowing the dataset down into one cohesive
      dataset, where the SMOTE process is used only with Myopia Score as the target, with the remaining
      null columns for CVS ans Astigmatism being filled in using KNN.
"""

import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE
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
        'genetics', 'age_first_rx'
    ]
    
    normalized_df = pd.DataFrame()
    for col in input_cols:
        normalized_df[col] = normalize_column(df[col])
    
    normalized_df["myopia_score"] = (
        df["myopia_level"] * (1 + df["refractive_worsening"] * 0.1)
    )

    normalized_df["cvs_score"] = (
        df["cvs_headache_strain"] * (1 + df["cvs_dry_eyes"] * 0.1)
    )

    normalized_df["astigmatism_score"] = (
        df["astigmatism_symptoms"] * (1 + df["myopia_level"] * 0.1)
    )
    
    return normalized_df

def apply_smote_for_ml(df, primary_target):
    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx'
    ]

    X = df[input_cols].copy()
    y_continuous = df[primary_target].copy()
    
    # Convert continuous scores to ordinal integer classes (0 to 4)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [0, 1, 2, 3, 4]
    y = pd.cut(y_continuous, bins=bins, labels=labels, include_lowest=True)
    y = pd.factorize(y)[0]

    print(f"  Class distribution before SMOTE (using {primary_target}):")
    print(f"  {pd.Series(y).value_counts().sort_index().to_dict()}")
    print(f"  Total samples: {len(df)}")

    min_class_size = pd.Series(y).value_counts().min()
    k_neighbors = max(1, min_class_size - 1)
    print(f"  Using k_neighbors={k_neighbors} for SMOTE")

    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"  Class distribution after SMOTE:")
    print(f"  {pd.Series(y_resampled).value_counts().sort_index().to_dict()}")
    print(f"  Total samples after SMOTE: {len(X_resampled)}")

    print(f"  Rounding synthetic values to valid ordinal ranges...")
    X_resampled = round_to_valid_values(X_resampled, df)
    print(f"  Rounding complete!")

    for col in input_cols:
        X_resampled[col] = np.clip(X_resampled[col], 0.0, 1.0)

    balanced_df = pd.DataFrame(X_resampled, columns=input_cols)
    balanced_df[primary_target] = y_resampled

    return balanced_df

def main():
    parser = argparse.ArgumentParser(description='Encode survey CSV to ordinal values.')
    parser.add_argument('--input_path', default='../OUTPUT.csv')
    parser.add_argument('--output_prefix', default='training_data_smote_')
    
    args = parser.parse_args()
 
    input_path  = Path(args.input_path)
    output_prefix = Path(args.output_prefix)

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not correlate to an adequate CSV file")
        return
    
    print("=" * 70)
    print("MACHINE LEARNING - SMOTE DATA AUGMENTATION")
    print("The prefix is:")
    print(output_prefix)
    print("=" * 70)
    
    print(f"\nLoading and normalizing survey responses...")
    df = load_and_normalize_inputs(input_path)
    print(f"Loaded {len(df)} survey responses with 15 normalized inputs")
    
    print(f"\nApplying SMOTE with target: Myopia Score (Current Nearsightedness and Prescription Change)...")
    myopia_balanced_df = apply_smote_for_ml(df, primary_target="myopia_score")
    computervs_balanced_df = apply_smote_for_ml(df, primary_target="cvs_score")
    astigmatism_balanced_df = apply_smote_for_ml(df, primary_target="astigmatism_score")
    
    print(f"\nSaving balanced dataset...")
    myopia_balanced_df.to_csv(str(output_prefix) + "myopia.csv", index=False)
    computervs_balanced_df.to_csv(str(output_prefix) + "computervs.csv", index=False)
    astigmatism_balanced_df.to_csv(str(output_prefix) + "astigmatism.csv", index=False)
    
    print(f"Saved individual datasets into respective CSV files")
    
    print("\n" + "=" * 70)
    print("SMOTE processing complete! Dataset ready for individual ML model training.")
    print("=" * 70)

if __name__ == "__main__":
    main()
