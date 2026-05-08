# For testing purposes for the model logic and initial data analysis without
# having to wait for the survey responses, we will be faking the actual data
# using Numpy and Pandas.

# Pandas - used for data manipulation and analysis
# Numpy - used for numerical computing and math

# Of course, the end results of the survey will be more spread out and randomised
# For the purpose of this segment, a more normal distribution is taken.
import numpy as np;
import pandas as pd;

# For purpose of consistency, this generates the exact same random outputs each time when
# testing this codebase.
np.random.seed(42)
N = 100;

# Each response here aligns to a question from the survey.
fake_responses = pd.DataFrame({
    # --- INPUT VARIABLES --- #
    'age' : np.random.randint(6, 45, N),
    'gender' : np.random.randint(0, 2, N),
    'daily_screen_time' : np.random.choice([1,2,3,4,5,6,7,8], N, p=[0.02, 0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.13]),
    'continuous_usage' : np.random.choice([1,2,3,4,5], N, p=[0.05, 0.15, 0.30, 0.30, 0.20]),
    'intensity': np.random.choice([1,2,3,4,5], N, p=[0.05, 0.15, 0.30, 0.30, 0.20]),
    'lighting': np.random.choice([1,2,3,4,5], N),
    'multi_device': np.random.choice([1,2,3,4,5], N, p=[0.10, 0.20, 0.30, 0.25, 0.15]),
    'phone_distance': np.random.choice([1,2,3,4,5], N),
    'monitor_distance': np.random.choice([1,2,3,4,5], N),
    'blue_light_filter': np.random.choice([1,2,3,4,5], N),
    'before_bed_usage': np.random.choice([1,2,3,4,5], N),
    'profession': np.random.choice([1,2,3,4,5], N, p=[0.05, 0.15, 0.30, 0.30, 0.20]),
    'outdoor_activity': np.random.choice([1,2,3,4,5], N, p=[0.20, 0.30, 0.25, 0.15, 0.10]),
    'genetics': np.random.choice([1,2,3,4,5], N, p=[0.20, 0.20, 0.20, 0.20, 0.20]),
    'age_first_rx': np.random.choice([0,6,8,10,12,14,16,18,20], N),
    # --- TARGET VARIABLES --- #
    'myopia_level': np.random.choice([0,1,2,3,4], N, p=[0.25, 0.30, 0.25, 0.15, 0.05]),
    'refractive_worsening': np.random.choice([0,1,2,3], N, p=[0.30, 0.35, 0.25, 0.10]),
    'cvs_headache_strain': np.random.choice([0,1,2,3,4], N),
    'cvs_dry_eyes': np.random.choice([0,1,2,3,4], N),
    'astigmatism_symptoms': np.random.choice([0,1,2,3,4], N, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
})

fake_responses.to_csv('fake_survey_responses.csv', index=False)