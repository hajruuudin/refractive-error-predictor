import pandas as pd;

# Since all values need to be between 0.0 and 1.0, we normalise its values
def normalise(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

def build_composite_results(data):
    scores = pd.DataFrame()

    # STRAIN INDEX = Weighted average of continous time and total screen time
    strain_screen = normalise(data['daily_screen_time'], 1, 8)
    strain_continuous = normalise(data['continuous_usage'], 1, 5)
    scores['strain_index'] = (0.70 * strain_screen + 0.65 * strain_continuous) / (0.70 + 0.65)

    # DIGITAL CONSUMPTION DETRIMENTS = Weighted average of all other digital consumption related effects
    phone_dist_risk   = 1 - normalise(data['phone_distance'],   1, 5)
    monitor_dist_risk = 1 - normalise(data['monitor_distance'], 1, 5)
    viewing_distance_risk = (phone_dist_risk + monitor_dist_risk) / 2  # average the two

    lighting_risk = normalise(data['lighting'], 1, 5)
    multi_risk = normalise(data['multi_device'], 1, 5)
    intensity_risk = normalise(data['intensity'], 1, 5)
    before_bed_risk = normalise(data['before_bed_usage'], 1, 5)
    filter_risk = 1 - normalise(data['blue_light_filter'], 1, 5)

    w_total = 0.75 + 0.55 + 0.50 + 0.60 + 0.60 + 0.35
    scores['digital_consumption_detriments'] = (
        0.75 * viewing_distance_risk +
        0.55 * lighting_risk         +
        0.50 * multi_risk            +
        0.60 * intensity_risk        +
        0.60 * before_bed_risk       +
        0.35 * filter_risk
    ) / w_total

    # LIFESTYLE EFFECTS = Profession and outdoor activity
    outdoor_risk = 1 - normalise(data['outdoor_activity'], 1, 5)
    profession_risk = normalise(data['profession'], 1, 5)

    w_total = 0.80 + 0.55
    scores['lifestyle_effects'] = (
        0.80 * outdoor_risk   +
        0.55 * profession_risk
    ) / w_total

    # --- 4. BIOLOGICAL FACTORS ---
    genetics_risk = normalise(data['genetics'], 1, 5)

    # age_first_rx: Earlier onset = higher risk for myopia.
    # 0 means no prescription. We treat 0 as the safest (no prescription at all),
    # and earlier non-zero ages as higher risk.
    # Approach: map 0→0 risk, 6→highest risk, 20→low risk
    def age_rx_risk(age):
        if age == 0:
            return 0.0 
        return 1 - normalise(min(max(age, 5), 25), 5, 25)
    age_rx_risk_series = data['age_first_rx'].apply(age_rx_risk)

    gender_risk = data['gender'] * 0.4
    age_risk = normalise(data['age'].clip(6, 45), 6, 45)

    w_total = 0.90 + 0.75 + 0.40 + 0.50
    scores['biological_factors'] = (
        0.90 * genetics_risk     +
        0.75 * age_rx_risk_series +
        0.40 * gender_risk       +
        0.50 * age_risk
    ) / w_total

    return scores;

def main():
    fake_responses = pd.read_csv('fake_survey_responses.csv')
    composite_scores = build_composite_results(fake_responses)
    pd.DataFrame(composite_scores)
    print(composite_scores.head())
    composite_scores.to_csv('composite_scores.csv')

    
if __name__ == '__main__':
    main()