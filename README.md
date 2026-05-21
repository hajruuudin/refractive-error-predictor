# Masters Thesis Codebase: ML Model for Eyesight Errors based on Digital Consumption Patterns (WIP)
Codebase for the machine learning &amp; data anlysis segment of a Masters thesis: Analysing the correlation &amp; creating a predictive model for determining the detrimental effects digital consumption patterns have on refractive errors &amp; vision health

# Workflow
As of writing this README.md, the idea of the repository remains as follows:
- The expected amount of survey respondants needs to reach at minimum 150 in order for the data to be effective. 100 is also a reach, however, at this low rate we can expect inacuracies with the end results
- For the purpose of setting up the environment and familiarising myself with the codebase, a fake survey response of roughly 500 is generated within the script `fake_survey_response_gen.py`. This script generates an identical result base to the survey with 15 regressor questions and 5 target responses, with the results expressed as ordinal numerical values.
- After data generation (or in the real-life scenario, after taking the actual survey results and generalising them to the ordinal values), we proceed with two different workflows:
    - **Machine Learning Workflow**: SMOTE is used to extend the original dataset into an equal minority class and majority class split, with options to increase the total number of results per-class. Currently, SMOTE only increases the minority class up until it matches the majority classes. SMOTE is also repeated three times, each time for a new target variable.
    - The target variables are compound into three compund target instead of being 5 individual targets. For example, nearsightedness and vision degradation make up the total MYOPIA score, which is used as a predictor.
    - Furthermore, all three of the CSV files for each individual target are used for training three ML models based on two different frameworks: Random Forest and XGBoost. Each of the Models gives a round estimate of the most important features as well as being tested using MAE, R2 and CV_MAE.
    - **Linear Equation Workflow:** Takes the same CSV from the fake responses and generates a linear equation. This is currently a work in progress.

_The weights, special cases, overall workflow and small adjustments to be made to the models will be specified later on. The basis of the repository stays the same. Last edit: Thu, 21st May_