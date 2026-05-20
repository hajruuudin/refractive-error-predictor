"""
PART 00: GENERATING FAKE SURVEY RESPONSES FOR ML MODEL TRAINING

For the purpose of testing and understanding of the code without having to wait for the survey results,
we will be generating the data at random using this script.
This script creates a CSV file which mimics the results of the actual survey inputs AFTER survey formatting.
By this, we mean to say that the survey itself gives categorical choices which need to be mapped to numerical inputs.

The distribution of the results of this fake survey is the same for all executions, which is set by the random seed of 42 (42 is arbitrary). This
is to ensure that testing the further framework can be done predicibly and that the end-results can stay consistent. This also means that, due to 
the randomness of the results, the actual findings the ML model makes should not be taken into account as those will not be representative of 
the actual distribution of the real-life results.

Furthermore, the responses are divided into two groups:
- Input variables: these are the regressors, or more commonly known as predictive indicators
- Output variables: these are the outputs we aim to predict MYO, AST and CVS with.
Do note that while there are three outcomes we predict, we do have 5 target questions. These 5 questions are aggregated into three results
instead of being just one questions per predictive goal. For example, Targets 1 and 2 are both for MYO, hence their average scores are
taken into account when predicting the likelyhood a user has to experience Myopic worsening.

The size of the CSV file is 200. This is adjustable via the N variables. Do note however that by the configuration of SMOTE, we require at least
3 samples per minority class for each question in order to generate new samples. Otherwise, the SMOTE process will fail. This is also adjustable within the 
machine_learning_smote.py script, however, a reduction in this parameter can induce an inaccurate sample creation.
"""
import numpy as np;
import pandas as pd;

np.random.seed(42) # Setting the seed of the Generator for the CSV file to be exactly the same each time
N = 200; # By default this generates 200 results.

fake_responses = pd.DataFrame({ # Each response here aligns to a question from the survey.
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
    'myopia_level': np.random.choice([0,1,2,3,4,5], N, p=[0.15, 0.20, 0.30, 0.20, 0.10, 0.05]),
    'refractive_worsening': np.random.choice([0,1,2,3], N, p=[0.30, 0.35, 0.25, 0.10]),
    'cvs_headache_strain': np.random.choice([0,1,2,3,4], N),
    'cvs_dry_eyes': np.random.choice([0,1,2,3,4], N),
    'astigmatism_symptoms': np.random.choice([0,1,2,3,4], N, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
})

fake_responses.to_csv('fake_survey_responses.csv', index=False)