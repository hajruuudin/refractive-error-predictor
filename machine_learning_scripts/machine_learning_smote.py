"""
SMOTE for Machine Learning Pipeline

For the ML model, we start with the raw 15 input variables (normalized 0-1) and 
apply SMOTE to balance the dataset across target variables. Unlike the linear model,
the ML model learns weights from data, so we need sufficient balanced samples.

Pipeline:
1. Load 500 fake survey responses
2. Normalize 15 input variables to 0-1
3. Create target variable for balancing (primary outcome)
4. Apply SMOTE to balance across target classes
5. Export balanced dataset for ML model training
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
import os

def normalize_column(col):
    """Normalize a column to 0-1 range"""
    min_val = col.min()
    max_val = col.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(col))
    return (col - min_val) / (max_val - min_val)

def get_valid_normalized_values(col):
    """
    Get all valid normalized values for an ordinal column.
    
    For ordinal variables, after normalization, only discrete values are valid.
    E.g., for 1-5 scale: [0.0, 0.25, 0.5, 0.75, 1.0]
    E.g., for 0-8 scale: [0.0, 0.125, 0.25, ..., 1.0]
    
    Args:
        col: Original (non-normalized) column with min/max values
    
    Returns:
        Array of valid normalized values for this column
    """
    min_val = col.min()
    max_val = col.max()
    
    if max_val == min_val:
        return np.array([0.5])
    
    # Generate all possible values from min to max
    possible_original_values = np.arange(min_val, max_val + 1)
    
    # Normalize them
    valid_normalized = (possible_original_values - min_val) / (max_val - min_val)
    
    return valid_normalized

def round_to_valid_values(synthetic_data, original_df):
    """
    Round SMOTE-generated synthetic values to nearest valid ordinal values.
    
    SMOTE interpolates between points, which can create invalid values.
    This function rounds each synthetic value to the nearest valid discrete value
    that could exist in the original data.
    
    Args:
        synthetic_data: DataFrame with synthetic values (0-1 normalized, may have invalid values)
        original_df: Original DataFrame to extract valid value ranges
    
    Returns:
        DataFrame with rounded synthetic data (valid ordinal values only)
    """
    rounded_data = synthetic_data.copy()
    
    # Get input columns (exclude targets)
    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx'
    ]
    
    # Need to denormalize, round to nearest valid value, then renormalize
    for col in input_cols:
        if col in synthetic_data.columns:
            # Get valid normalized values for this column
            valid_values = get_valid_normalized_values(original_df[col])
            
            # For each synthetic value, find nearest valid value
            for idx in rounded_data.index:
                synthetic_val = rounded_data.loc[idx, col]
                # Find nearest valid value
                nearest_idx = np.argmin(np.abs(valid_values - synthetic_val))
                rounded_data.loc[idx, col] = valid_values[nearest_idx]
    
    return rounded_data

def load_and_normalize_inputs(csv_file):
    """
    Load survey responses and normalize the 15 input variables.
    
    Args:
        csv_file: Path to fake_survey_responses.csv
    
    Returns:
        DataFrame with normalized inputs (0-1) and target variables
    """
    df = pd.read_csv(csv_file)
    
    # Define the 15 input variables
    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx'
    ]
    
    # Define the 5 target variables
    target_cols = [
        'myopia_level', 'refractive_worsening', 'cvs_headache_strain',
        'cvs_dry_eyes', 'astigmatism_symptoms'
    ]
    
    # Normalize input variables
    normalized_df = pd.DataFrame()
    for col in input_cols:
        normalized_df[col] = normalize_column(df[col])
    
    # Add target variables (not normalized, kept as-is)
    for col in target_cols:
        normalized_df[col] = df[col]
    
    return normalized_df

def apply_smote_for_ml(df, original_df, primary_target='myopia_level'):
    """
    Apply SMOTE to balance dataset across primary target variable.
    
    For synthetic samples created by SMOTE, use KNN to find k-nearest original samples
    and assign their target values (averaged). This preserves realistic target relationships
    without baking in assumptions from the linear model.
    
    Args:
        df: DataFrame with normalized inputs (0-1) and targets
        original_df: Original (non-normalized) DataFrame for KNN reference
        primary_target: Which target to balance by (myopia_level, astigmatism_symptoms, etc.)
    
    Returns:
        SMOTE-balanced DataFrame with realistic target values via KNN
    """
    # Store original (non-normalized) data for reference
    # Define input and target columns
    input_cols = [
        'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
        'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
        'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
        'genetics', 'age_first_rx'
    ]
    target_cols = [
        'myopia_level', 'refractive_worsening', 'cvs_headache_strain',
        'cvs_dry_eyes', 'astigmatism_symptoms'
    ]
    
    X = df[input_cols].copy()
    y = df[primary_target].copy()
    X_original = original_df[input_cols].copy()
    
    print(f"  Class distribution before SMOTE (using {primary_target}):")
    print(f"  {y.value_counts().sort_index().to_dict()}")
    print(f"  Total samples: {len(df)}")
    
    # Dynamically set k_neighbors based on smallest class
    min_class_size = y.value_counts().min()
    k_neighbors = max(1, min_class_size - 1)
    print(f"  Using k_neighbors={k_neighbors} for SMOTE")
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"  Class distribution after SMOTE:")
    print(f"  {pd.Series(y_resampled).value_counts().sort_index().to_dict()}")
    print(f"  Total samples after SMOTE: {len(X_resampled)}")
    
    # Round synthetic values to valid ordinal values
    print(f"  Rounding synthetic values to valid ordinal ranges...")
    X_resampled = round_to_valid_values(X_resampled, original_df)
    print(f"  Rounding complete!")
    
    # Enforce constraints on normalized inputs (keep in 0-1 range)
    for col in input_cols:
        X_resampled[col] = np.clip(X_resampled[col], 0.0, 1.0)
    
    # Use KNN to assign target values from nearest original samples
    print(f"  Assigning target values via KNN to nearest original samples...")
    n_neighbors = min(5, len(X_original))  # Use up to 5 nearest neighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(X_original)
    
    # Find k-nearest neighbors for each synthetic sample
    distances, indices = knn.kneighbors(X_resampled)
    
    # Initialize target arrays
    target_arrays = {target: np.zeros(len(X_resampled)) for target in target_cols}
    
    # For each synthetic sample, average the target values of its k-nearest neighbors
    for i in range(len(X_resampled)):
        nearest_indices = indices[i]
        for target in target_cols:
            nearest_values = original_df.iloc[nearest_indices][target].values
            target_arrays[target][i] = np.mean(nearest_values)
    
    print(f"  Target values assigned from {n_neighbors}-nearest neighbors!")
    
    # Reconstruct DataFrame with inputs and all targets
    balanced_df = pd.DataFrame(X_resampled, columns=input_cols)
    for target in target_cols:
        balanced_df[target] = target_arrays[target]
    
    return balanced_df

def main():
    """
    Generate SMOTE-balanced dataset for ML model training.
    """
    # Paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    input_csv = os.path.join(project_root, 'fake_survey_responses.csv')
    output_csv = os.path.join(os.path.dirname(__file__), 'training_data_smote.csv')
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found")
        return
    
    print("=" * 70)
    print("MACHINE LEARNING - SMOTE DATA AUGMENTATION")
    print("=" * 70)
    
    print(f"\nLoading and normalizing survey responses...")
    df = load_and_normalize_inputs(input_csv)
    print(f"Loaded {len(df)} survey responses with 15 normalized inputs")
    
    # Also load original for KNN reference
    original_df = pd.read_csv(input_csv)
    
    print(f"\nApplying SMOTE with KNN-based target assignment...")
    balanced_df = apply_smote_for_ml(df, original_df, primary_target='myopia_level')
    
    print(f"\nSaving balanced dataset...")
    balanced_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    
    print("\n" + "=" * 70)
    print("SMOTE processing complete! Dataset ready for ML model training.")
    print("=" * 70)

if __name__ == "__main__":
    main()
