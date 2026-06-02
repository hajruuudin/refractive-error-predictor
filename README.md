# Masters Thesis Codebase: ML Model for Eyesight Errors based on Digital Consumption Patterns (WIP)
Codebase for the machine learning &amp; data anlysis segment of a Masters thesis: Analysing the correlation &amp; creating a predictive model for determining the detrimental effects digital consumption patterns have on refractive errors &amp; vision health

# Workflow
The following paragraph explains how the codebase is intended to be used.
- The expected amount of survey respondants needs to reach at minimum 150 in order for the data to be effective. 100 is also a reach, however, at this low rate we can expect inacuracies with the end results
- For testing purposes and creation of the workflow, a fake survey response generator (within the file `fake_survey_response_gen.py`) is present within the repository root. This file creates 500 randomly generated samples in order to test the code syntax and workflow. Note that the results of the models will be innacurate simply because this approach generates random samples.
- The survey itself is present on this link:
- In case the survey is not available, the structure of the survey responses is the same as in the fake response geenrator. It is also available within the script `survey_data_transformation.py` where the original responses are converted from categorical, worded responses into ordinal, numerical values.
- After inserting the survey responses, one of the 5 SMOTE variants and 6 TARGET Equations needs to be selected. The target equations are present within the `machine_learning_scripts/target_equations.py` file, and each of the SMOTE variants is also located within the same directory.
- Each SMOTE variant has all 6 target equations coded. The equations of choice needs to be uncommented, with the other being commented or deleted.
- After setting the SMOTE file, running it creates three CSV files, each correlating to one of the three targets (MYOPIA, COMPUTER VISION SYNDROME &amp; ASTIGMATISM).
- Within the same directory, two files exist for two different models:
    - RandomForest: `machine_learning_randomForest.py`
    - XgBoost: `machine_learning_xgBoost.py`
- Both algorithms follow the default setup for their hyperparameters. These are modifiable, however, the performance largely depends more on the number of survey respondants rather than the algorithm setup. The setup creates an 80/20 split for testing and training.

