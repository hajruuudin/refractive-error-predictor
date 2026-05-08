import pandas as pd

def build_targets(df):
    targets = pd.DataFrame()

    # --- MYOPIA RISK (0=None, 1=Low, 2=Moderate, 3=High) ---
    # Combines current prescription level + worsening trend
    myopia_score = df['myopia_level'] + df['refractive_worsening']
    # myopia_score now ranges from 0 to 7
    targets['myopia_risk'] = pd.cut(
        myopia_score,
        bins=[-1, 1, 3, 5, 7],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # --- CVS RISK (0=None, 1=Low, 2=Moderate, 3=High) ---
    # Combines headache/strain frequency + dry eyes frequency
    cvs_score = df['cvs_headache_strain'] + df['cvs_dry_eyes']
    # cvs_score now ranges from 0 to 8
    targets['cvs_risk'] = pd.cut(
        cvs_score,
        bins=[-1, 1, 3, 5, 8],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # --- ASTIGMATISM RISK (0=None, 1=Low, 2=Moderate, 3=High) ---
    # Based on ghosting/symptom frequency + genetics contribution
    astigmatism_score = df['astigmatism_symptoms'] + (df['genetics'] - 1)
    # astigmatism_score ranges from 0 to 8
    targets['astigmatism_risk'] = pd.cut(
        astigmatism_score,
        bins=[-1, 1, 3, 5, 8],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return targets

def main():
    fake_responses = pd.read_csv('fake_survey_responses.csv')
    targets = build_targets(fake_responses)
    pd.DataFrame(targets)
    targets.to_csv("target_scores.csv")

if __name__ == "__main__":
    main()