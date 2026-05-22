
import argparse
import csv
import sys
from pathlib import Path
 
# ---------------------------------------------------------------------------
# Ordinal mapping dictionaries
# ---------------------------------------------------------------------------

# Q1 — age  (raw int)

# Q2 — gender  (0–1)
 
# Q3  — daily_screen_time  (1–8)
DAILY_SCREEN_TIME = {
    'Less than 1 hour':   1,
    '1 to 2 hours':       2,
    '2 to 3 hours':       3,
    '3 to 4 hours':       4,
    '4 to 5 hours':       5,
    '5 to 6 hours':       6,
    '6 to 7 hours':       7,
    'More than 7 hours':  8,
}
 
# Q4  — continuous_usage  (1–5)
CONTINUOUS_USAGE = {
    '10 to 20 minutes':     1,
    '20 to 30 minutes':     2,
    '30 to 40 minutes':     3,
    '40 to 60 minutes':     4,
    'More than 60 minutes': 5,
}
 
# Q5  — intensity  (1–5)
INTENSITY = {
    'Low focus — occasional glances, brief checks':                    1,
    'Below average — standard scrolling and messaging':                2,
    'Average — general content consumption with occasional reading':   3,
    'Above average — frequent high-focus tasks, near-work required':   4,
    'Very high intensity — demanding job with heavy digital content':  5,
}
 
# Q6  — lighting  (1–5)  1 = brightest, 5 = darkest
LIGHTING = {
    'Almost exclusively bright/high-brightness environments': 1,
    'Mostly bright, occasional dim lighting':                 2,
    'Even mix of bright and dark environments':               3,
    'Mostly dim, occasional bright environment':              4,
    'Almost exclusively in dark/dim ambient lighting':        5,
}
 
# Q7  — multi_device  (1–5)
MULTI_DEVICE = {
    'Never (I focus on one screen only).':                                               1,
    'Rarely (Only for quick checks and ocassional information glances).':                2,
    'Occasionally (Daily for short periods).':                                           3,
    'Often (I use my phone and laptop often in conjunction).':                           4,
    'Almost always (my day-to-day tasks require me to use two or more devices simultaneously).': 5,
}
 
# Q8  — phone_distance  (1–5)  1 = closest, 5 = farthest
PHONE_DISTANCE = {
    'Very Close (Less than 15cm / Really close)':         1,
    'Close (15cm to 25cm / Near work)':                   2,
    "Standard (25cm to 35cm / Bent Arm's length)":        3,
    'Semi-far (35cm to 50cm / Arms Length)':              4,
    'Far (More than 50cm / More than Arms length)':       5,
}
 
# Q9  — monitor_distance  (1–5)  1 = closest, 5 = farthest
MONITOR_DISTANCE = {
    'Very Close, (Less than 20cm / High focus & intensity)': 1,
    'Close (20cm to 35cm / Near work)':                      2,
    "Medium (35cm to 50cm / Arm's length)":                  3,
    'Standard (50cm to 1m / Desktop monitor)':               4,
    'Far (More than 1m / Large TV or Projector)':            5,
}
 
# Q10 — blue_light_filter  (1–5)
BLUE_LIGHT_FILTER = {
    'Never':                           1,
    'Rarely (Only late at night)':     2,
    'Occasionally':                    3,
    'Frequently (Most of the day)':    4,
    'Always (Enabled on all devices at all times)': 5,
}
 
# Q11 — before_bed_usage  (1–5)
BEFORE_BED_USAGE = {
    'Never':                        1,
    'Rarely (1-2 nights a week)':   2,
    'Sometimes (3-4 nights a week)':3,
    'Often (5-6 nights a week)':    4,
    'Almost always (every night)':  5,
}
 
# Q12 — profession  (1–5)
PROFESSION = {
    'No digital usage (Manual labor / Outdoor work)':          1,
    'Low usage (Occasional check-ins / Retail)':               2,
    'Moderate usage (Office work / Standard admin)':           3,
    'Moderate usage (Oace work / Standard admin)':             3,   # typo variant in data
    'High usage (Student / Extensive research)':               4,
    'Extreme usage (Software Engineering / Data Analysis / Pro Gaming)': 5,
}
 
# Q13 — outdoor_activity  (1–5)
OUTDOOR_ACTIVITY = {
    'Less than 30 minutes':    1,
    '30 minutes to 1 hour':    2,
    '1 to 2 hours':            3,
    '2 to 3 hours':            4,
    'More than 3 hours':       5,
}
 
# Q14 — genetics  (1–5)
GENETICS = {
    'No history / Neither parent':       1,
    'One parent (Mild prescription)':    2,
    'One parent (Strong prescription)':  3,
    'Both parents (Mild to Moderate)':   4,
    'Both parents (Strong prescriptions)': 5,
}

# Q15 — age_first_rx  (raw int)
 
# Q16 — myopia_level  (0–5)  ; "Unsure" → -1 (handle downstream)
MYOPIA_LEVEL = {
    '0.0 to -0.5 (No prescription or very mild blur — distant objects are mostly clear)': 0,
    '-0.5 to -1.5 (Mild nearsightedness — distant signs and faces begin to blur noticeably': 1,
    '-1.5 to -3.0 (Moderate nearsightedness — TV across a room or a whiteboard is difficult without correction)': 2,
    "-3.0 to -4.5 (Strong nearsightedness — anything beyond arm's length is significantly blurred)": 3,
    '-4.5 and below (Severe nearsightedness — functional vision without correction is very limited)': 4,
    "Unsure (I have a prescription but don't remember the value, or I have a different type of error)": -1,
}
 
# Q17 — refractive_worsening  (0–3)
REFRACTIVE_WORSENING = {
    'Stable (No change in prescription or clarity)':                                           0,
    'Worsened Slightly (Minor change / Slight blurrienes for objects in the distance)':        1,
    'Worsened Moderately (Noticeable blurriness / Need to focus hard to see distant objects)': 2,
    'Worsened Significantly (Required much stronger glasses / Almost impossible to see far clearly)': 3,
}
 
# Q18 — cvs_headache_strain  (0–4)
CVS_HEADACHE = {
    'Never':                             0,
    'Rarely (Once or twice a month)':    1,
    'Occasionally (Once or twice a week)':2,
    'Frequently (3 to 5 times a week)':  3,
    'Almost Daily':                      4,
}
 
# Q19 — cvs_dry_eyes  (0–4)
CVS_DRY_EYES = {
    'Never':                             0,
    'Rarely (Once or twice a month)':    1,
    'Occasionally (Once or twice a week)':2,
    'Frequently (3 to 5 times a week)':  3,
    'Almost Daily':                      4,
}
 
# Q20 — astigmatism_symptoms  (0–4)
ASTIGMATISM_SYMPTOMS = {
    'Never (Clear edges)':                        0,
    'Rarely (Only when eyes are very tired)':     1,
    'Occasionally (In low-light conditions)':     2,
    'Frequently (Noticeable on most text)':       3,
    'Constantly (Persistent distortion regardless of light)': 4,
}
 
# ---------------------------------------------------------------------------
# Column index → (output name, mapping dict)
# age      = col 1  → raw int
# gender   = col 2  → Male=0 / Female=1
# age_first_rx = col 15 → raw int
# ---------------------------------------------------------------------------
 
COLUMN_MAP = {
    3:  ('daily_screen_time',    DAILY_SCREEN_TIME),
    4:  ('continuous_usage',     CONTINUOUS_USAGE),
    5:  ('intensity',            INTENSITY),
    6:  ('lighting',             LIGHTING),
    7:  ('multi_device',         MULTI_DEVICE),
    8:  ('phone_distance',       PHONE_DISTANCE),
    9:  ('monitor_distance',     MONITOR_DISTANCE),
    10: ('blue_light_filter',    BLUE_LIGHT_FILTER),
    11: ('before_bed_usage',     BEFORE_BED_USAGE),
    12: ('profession',           PROFESSION),
    13: ('outdoor_activity',     OUTDOOR_ACTIVITY),
    14: ('genetics',             GENETICS),
    16: ('myopia_level',         MYOPIA_LEVEL),
    17: ('refractive_worsening', REFRACTIVE_WORSENING),
    18: ('cvs_headache_strain',  CVS_HEADACHE),
    19: ('cvs_dry_eyes',         CVS_DRY_EYES),
    20: ('astigmatism_symptoms', ASTIGMATISM_SYMPTOMS),
}
 
OUTPUT_COLUMNS = [
    'age', 'gender',
    'daily_screen_time', 'continuous_usage', 'intensity',
    'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
    'blue_light_filter', 'before_bed_usage', 'profession',
    'outdoor_activity', 'genetics', 'age_first_rx',
    'myopia_level', 'refractive_worsening',
    'cvs_headache_strain', 'cvs_dry_eyes', 'astigmatism_symptoms',
]
 
 
def encode_gender(value: str) -> int:
    v = value.strip().lower()
    if v == 'male':
        return 0
    if v == 'female':
        return 1
    raise ValueError(f"Unknown gender value: '{value}'")
 
 
def encode_row(row: list[str], row_num: int) -> dict:
    encoded = {}
    warnings = []
 
    # --- Age (col 1) ---
    try:
        encoded['age'] = int(float(row[1].strip()))
    except ValueError:
        encoded['age'] = None
        warnings.append(f"  Row {row_num}: could not parse age '{row[1]}'")
 
    # --- Gender (col 2) ---
    try:
        encoded['gender'] = encode_gender(row[2])
    except ValueError as e:
        encoded['gender'] = None
        warnings.append(f"  Row {row_num}: {e}")
 
    # --- Categorical columns ---
    for col_idx, (col_name, mapping) in COLUMN_MAP.items():
        raw = row[col_idx].strip()
        if raw in mapping:
            encoded[col_name] = mapping[raw]
        else:
            encoded[col_name] = None
            warnings.append(f"  Row {row_num} [{col_name}]: unrecognised value '{raw}'")
 
    # --- Age of first prescription (col 15) ---
    try:
        val = int(float(row[15].strip()))
        encoded['age_first_rx'] = val
    except ValueError:
        encoded['age_first_rx'] = None
        warnings.append(f"  Row {row_num}: could not parse age_first_rx '{row[15]}'")
 
    for w in warnings:
        print(w, file=sys.stderr)
 
    return encoded
 
 
def main():
    parser = argparse.ArgumentParser(description='Encode survey CSV to ordinal values.')
    parser.add_argument('--input',  default='Digital_Consumption_Habits___Vision_Health_Research_Survey.csv')
    parser.add_argument('--output', default='survey_encoded.csv')
    args = parser.parse_args()
 
    input_path  = Path(args.input)
    output_path = Path(args.output)
 
    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")
 
    with input_path.open(newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        header = next(reader)  # skip original header
        rows   = list(reader)
 
    print(f"Read {len(rows)} responses from '{input_path}'")
 
    encoded_rows = []
    for i, row in enumerate(rows, start=2):   # row 2 = first data row
        if len(row) < 21:
            print(f"  Row {i}: skipped (only {len(row)} columns)", file=sys.stderr)
            continue
        encoded_rows.append(encode_row(row, i))
 
    # --- Write output ---
    with output_path.open('w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(encoded_rows)
 
    # --- Summary ---
    null_counts = {col: sum(1 for r in encoded_rows if r.get(col) is None)
                   for col in OUTPUT_COLUMNS}
    total = len(encoded_rows)
    print(f"\nEncoded {total} rows → '{output_path}'")
    print("\nNull / unmatched counts per column:")
    for col, count in null_counts.items():
        flag = '  ⚠' if count > 0 else ''
        print(f"  {col:<25} {count:>3} / {total}{flag}")
 
    unsure_count = sum(1 for r in encoded_rows if r.get('myopia_level') == -1)
    if unsure_count:
        print(f"\n  Note: {unsure_count} respondent(s) answered 'Unsure' for myopia_level → encoded as -1.")
        print("        Filter or impute these rows before modelling.")
 
 
if __name__ == '__main__':
    main()
