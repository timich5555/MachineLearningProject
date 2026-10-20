
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "output/phq9_scored.csv"
OUTPUT_DIR = PROJECT_DIR / "output"

DATASET_FILE = OUTPUT_DIR / "model_dataset.csv"
DICTIONARY_FILE = OUTPUT_DIR / "feature_dictionary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def to_numeric(series):
    """Convert a column to numeric values; invalid values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def clean_byte_string(value):
    """
    Convert values such as b'W', b'D', b'M', b'Y'
    and literal strings like "b'W'" to D, W, M, Y.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]

    if value.startswith('b"') and value.endswith('"'):
        value = value[2:-1]

    value = value.strip().upper()

    if value in ("", "NAN", "NONE"):
        return np.nan

    return value


def weekly_frequency(frequency, unit):
    """
    Convert activity frequency into sessions per week.

    D = day
    W = week
    M = month
    Y = year
    """
    if pd.isna(frequency) or frequency < 0:
        return np.nan

    # Zero activity is a valid zero even when the unit is absent.
    if frequency == 0:
        return 0.0

    if pd.isna(unit):
        return np.nan

    unit = clean_byte_string(unit)

    conversion = {
        "D": 7.0,
        "W": 1.0,
        "M": 12.0 / 52.0,
        "Y": 1.0 / 52.0,
    }

    if unit not in conversion:
        return np.nan

    return float(frequency) * conversion[unit]


def activity_minutes_per_week(frequency, unit, duration):
    """
    Estimate minutes of leisure-time activity per week.

    If frequency is zero, return zero.
    If frequency is positive but unit or duration is unavailable,
    return NaN rather than inventing a value.
    """
    sessions = weekly_frequency(frequency, unit)

    if pd.isna(sessions):
        return np.nan

    if sessions == 0:
        return 0.0

    if pd.isna(duration) or duration <= 0:
        return np.nan

    return sessions * float(duration)


def annual_frequency(value):
    """
    Approximate the ALQ121 / ALQ142 frequency categories
    as number of occasions per year.

    These are estimates based on survey response categories,
    not exact observed counts.
    """
    if pd.isna(value):
        return np.nan

    mapping = {
        0: 0.0,       # Never
        1: 365.0,     # Every day
        2: 330.0,     # Almost every day
        3: 182.0,     # 3-4 times per week
        4: 104.0,     # Twice a week
        5: 52.0,      # Once a week
        6: 30.0,      # 2-3 times a month
        7: 12.0,      # About once a month
        8: 9.0,       # 7-11 times in the past year
        9: 4.5,       # 3-6 times in the past year
        10: 1.0,      # 1-2 times in the past year
    }

    return mapping.get(float(value), np.nan)


def activity_category(series):
    """
    Divide valid activity values into approximate thirds.

    Missing values remain missing.
    Tied quantile boundaries can result in empty categories;
    this is preferable to splitting identical values arbitrarily.
    """
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.dropna()

    if valid.empty:
        return result

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    result.loc[valid.index[valid <= q1]] = "low"

    if q2 > q1:
        middle = valid.index[(valid > q1) & (valid <= q2)]
        result.loc[middle] = "medium"

    if valid.max() > q2:
        high = valid.index[valid > q2]
        result.loc[high] = "high"

    return result


def require_columns(df, required_columns):
    """Stop with a clear error if required source columns are absent."""
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise KeyError(
            "The input file is missing required columns:\n"
            + ", ".join(missing)
            + "\nCheck that the correct merged/scored CSV is being used."
        )


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n\n"
        "Place phq9_scored.csv in the same folder as prepare_features.py."
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("DATA PREPARATION")
print("=" * 65)
print(f"Input file: {INPUT_FILE}")
print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "SEQN",
    "PHQ9_total",

    # Demographics
    "RIDAGEYR",
    "RIAGENDR",
    "DMDEDUC2",
    "DMDMARTZ",
    "INDFMPIR",
    "DMDHHSIZ",

    # Sleep
    "SLD012",
    "SLD013",

    # Smoking
    "SMQ020",
    "SMQ040",

    # Alcohol
    "ALQ121",
    "ALQ130",
    "ALQ142",

    # Physical activity
    "PAD790Q",
    "PAD790U",
    "PAD800",
    "PAD810Q",
    "PAD810U",
    "PAD820",
    "PAD680",

    # Food security and financial resources
    "FSDAD",
    "INQ300",
]

require_columns(df, required_columns)


# ============================================================
# 5. CONVERT NUMERIC SOURCE COLUMNS
# ============================================================

numeric_columns = [
    "SEQN",
    "PHQ9_total",
    "RIDAGEYR",
    "RIAGENDR",
    "DMDEDUC2",
    "DMDMARTZ",
    "INDFMPIR",
    "DMDHHSIZ",
    "SLD012",
    "SLD013",
    "SMQ020",
    "SMQ040",
    "ALQ121",
    "ALQ130",
    "ALQ142",
    "PAD790Q",
    "PAD800",
    "PAD810Q",
    "PAD820",
    "PAD680",
    "FSDAD",
    "INQ300",
]

for column in numeric_columns:
    df[column] = to_numeric(df[column])


# ============================================================
# 6. TARGET: PHQ-9 TOTAL SCORE
# ============================================================

# PHQ9_total is expected to have been calculated from DPQ010-DPQ090.
# We retain only rows with a valid target.
#
# DPQ special codes 7 and 9 must already have been handled when
# PHQ9_total was created. DPQ100 is not used as a predictor.

rows_before_target_filter = len(df)

df = df.loc[
    df["PHQ9_total"].notna()
    & df["PHQ9_total"].between(0, 27)
].copy()

print("\nTARGET FILTER")
print(f"Rows before filtering: {rows_before_target_filter}")
print(f"Rows with valid PHQ9_total: {len(df)}")
print(f"Rows excluded: {rows_before_target_filter - len(df)}")


# ============================================================
# 7. DEMOGRAPHIC FEATURES
# ============================================================

# RIDAGEYR:
# NHANES top-codes ages 80 and above as 80.
# Keep the supplied value; do not interpret 80 as an exact age
# for every participant aged 80 or older.

# RIAGENDR:
# 1 = Male, 2 = Female.
# Keep as a categorical code; no arbitrary numeric scaling here.

# DMDEDUC2:
# 1-5 are valid categories for adults aged 20+.
# 7 = Refused; 9 = Don't know.
df["DMDEDUC2"] = df["DMDEDUC2"].replace({
    7: np.nan,
    9: np.nan,
})
df.loc[~df["DMDEDUC2"].isin([1, 2, 3, 4, 5]), "DMDEDUC2"] = np.nan

# DMDMARTZ:
# 1 = Married/living with partner
# 2 = Widowed/divorced/separated
# 3 = Never married
# 77 = Refused; 99 = Don't know.
df["DMDMARTZ"] = df["DMDMARTZ"].replace({
    77: np.nan,
    99: np.nan,
})
df.loc[~df["DMDMARTZ"].isin([1, 2, 3]), "DMDMARTZ"] = np.nan

# INDFMPIR:
# Family income-to-poverty ratio.
# 5.00 represents 5.00 or greater in the released variable.
# Keep numeric values as supplied.

# DMDHHSIZ:
# 1-6 are exact household sizes; 7 represents 7 or more.
# Keep the source coding.


# ============================================================
# 8. SLEEP
# ============================================================

# SLD012: usual sleep hours on weekdays/workdays.
# SLD013: usual sleep hours on weekends/non-workdays.
#
# In the 2021-2023 NHANES cycle, the valid derived values include:
# 2 = less than 3 hours
# 3-13.5 = reported/derived hours
# 14 = 14 hours or more
#
# These variables do not use 7/9 or 77/99 as special codes.
# Do not remove 2 or 14: they are valid categories.

valid_sleep_values = (
    df["SLD012"].between(2, 14)
    & df["SLD013"].between(2, 14)
)

df["average_sleep_hours"] = np.nan

df.loc[valid_sleep_values, "average_sleep_hours"] = (
    5 * df.loc[valid_sleep_values, "SLD012"]
    + 2 * df.loc[valid_sleep_values, "SLD013"]
) / 7


# ============================================================
# 9. SMOKING
# ============================================================

# SMQ020:
# Ever smoked at least 100 cigarettes in lifetime?
# 1 = Yes, 2 = No, 7 = Refused, 9 = Don't know.

df["SMQ020"] = df["SMQ020"].replace({
    1: 1,
    2: 0,
    7: np.nan,
    9: np.nan,
})
df.loc[~df["SMQ020"].isin([0, 1]), "SMQ020"] = np.nan

# SMQ040:
# Current smoking frequency among respondents routed to this item.
# 1 = Every day
# 2 = Some days
# 3 = Not at all
# 7 = Refused
# 9 = Don't know
#
# Structural missingness from survey routing is retained as missing.
df["SMQ040"] = df["SMQ040"].replace({
    7: np.nan,
    9: np.nan,
})
df.loc[~df["SMQ040"].isin([1, 2, 3]), "SMQ040"] = np.nan


# ============================================================
# 10. ALCOHOL
# ============================================================

# Save original ALQ121 before replacing it with annualized values.
original_alq121 = df["ALQ121"].copy()

# ALQ121:
# Frequency of alcohol consumption in the past year.
# Special codes 77/99 and other unmapped values become NaN.
df["ALQ121"] = original_alq121.map(annual_frequency)

# ALQ130:
# Average number of drinks on drinking days.
# 1-14 = reported values; 15 = 15 or more.
# 777 = Refused; 999 = Don't know.
df["ALQ130"] = df["ALQ130"].replace({
    777: np.nan,
    999: np.nan,
})

# Keep only valid values from 1 through 15 before applying
# the special rule for respondents who never drink.
df.loc[~df["ALQ130"].between(1, 15), "ALQ130"] = np.nan

# If the original ALQ121 says "never", the average number of
# drinks on drinking days is set to zero for consistency.
no_alcohol = original_alq121 == 0
df.loc[no_alcohol, "ALQ130"] = 0

# IMPORTANT FIX:
# Check the converted ALQ121, not the original column's NaNs.
# This also catches special source codes 77/99 and unmapped values.
unknown_alcohol = df["ALQ121"].isna()
df.loc[unknown_alcohol, "ALQ130"] = np.nan

# ALQ142:
# Frequency of heavy drinking occasions during the past year.
# Convert the frequency categories using the same annualized mapping.
df["ALQ142"] = df["ALQ142"].map(annual_frequency)


# ============================================================
# 11. PHYSICAL ACTIVITY
# ============================================================

# NHANES PAQ_L measures leisure-time physical activity here,
# not all physical activity from every domain.
#
# PAD790Q / PAD790U / PAD800:
# Moderate leisure-time activity frequency, unit, and minutes.
#
# PAD810Q / PAD810U / PAD820:
# Vigorous leisure-time activity frequency, unit, and minutes.
#
# PAD680:
# Minutes spent sitting on a typical day.
#
# The documented refusal / don't-know codes for numeric PAQ fields
# include 7777 and 9999.

pa_numeric_columns = [
    "PAD790Q",
    "PAD800",
    "PAD810Q",
    "PAD820",
    "PAD680",
]

for column in pa_numeric_columns:
    df[column] = df[column].replace({
        7777: np.nan,
        9999: np.nan,
    })

# Clean the unit fields, including literal values such as "b'W'".
df["PAD790U"] = df["PAD790U"].map(clean_byte_string)
df["PAD810U"] = df["PAD810U"].map(clean_byte_string)

# Negative frequencies/durations are invalid.
for column in ["PAD790Q", "PAD800", "PAD810Q", "PAD820", "PAD680"]:
    df.loc[df[column] < 0, column] = np.nan

# Calculate weekly moderate activity minutes.
df["moderate_activity_min_week"] = [
    activity_minutes_per_week(freq, unit, duration)
    for freq, unit, duration in zip(
        df["PAD790Q"],
        df["PAD790U"],
        df["PAD800"],
    )
]

# Calculate weekly vigorous activity minutes.
df["vigorous_activity_min_week"] = [
    activity_minutes_per_week(freq, unit, duration)
    for freq, unit, duration in zip(
        df["PAD810Q"],
        df["PAD810U"],
        df["PAD820"],
    )
]

# Sitting time in minutes per typical day.
df["sedentary_minutes"] = df["PAD680"]


# ============================================================
# 12. FOOD SECURITY
# ============================================================

# FSDAD:
# 1 = Full food security among adults
# 2 = Marginal food security among adults
# 3 = Low food security among adults
# 4 = Very low food security among adults
#
# Missing values remain missing. Only categories 1-4 are retained.

df.loc[~df["FSDAD"].isin([1, 2, 3, 4]), "FSDAD"] = np.nan


# ============================================================
# 13. FINANCIAL RESOURCES
# ============================================================

# INQ300:
# Question wording: more than $5,000 in savings/cash assets.
# 1 = Yes
# 2 = No
# 7 = Refused
# 9 = Don't know

df["INQ300"] = df["INQ300"].replace({
    1: 1,
    2: 0,
    7: np.nan,
    9: np.nan,
})

df.loc[~df["INQ300"].isin([0, 1]), "INQ300"] = np.nan


# ============================================================
# 14. BUILD ALTERNATIVE ACTIVITY CATEGORIES
# ============================================================

# Categories are created after filtering to valid target rows.
# The thresholds are approximate thirds of the current analytic
# sample, including valid zero values.
#
# For a rigorous train/test ML workflow, learn these thresholds
# using only the training data, then apply them to validation/test
# data. Do not calculate category thresholds from the full dataset
# before a train/test split.

df["moderate_activity_category"] = activity_category(
    df["moderate_activity_min_week"]
)

df["vigorous_activity_category"] = activity_category(
    df["vigorous_activity_min_week"]
)


# ============================================================
# 15. SELECT FINAL FEATURES
# ============================================================

feature_columns = [
    "RIDAGEYR",
    "RIAGENDR",
    "DMDEDUC2",
    "DMDMARTZ",
    "INDFMPIR",
    "DMDHHSIZ",
    "average_sleep_hours",
    "SMQ020",
    "SMQ040",
    "ALQ121",
    "ALQ130",
    "ALQ142",
    "moderate_activity_min_week",
    "vigorous_activity_min_week",
    "sedentary_minutes",
    "FSDAD",
    "INQ300",
    "moderate_activity_category",
    "vigorous_activity_category",
]

final_columns = [
    "SEQN",
    "PHQ9_total",
] + feature_columns

dataset = df[final_columns].copy()


# ============================================================
# 16. FEATURE DICTIONARY
# ============================================================

feature_descriptions = {
    "SEQN": (
        "NHANES participant identifier; not a predictor."
    ),
    "PHQ9_total": (
        "Target: sum of the nine PHQ-9 symptom items, "
        "valid range 0-27."
    ),
    "RIDAGEYR": (
        "Age in years; NHANES top-codes ages 80 and above as 80."
    ),
    "RIAGENDR": (
        "Sex: 1 = male, 2 = female."
    ),
    "DMDEDUC2": (
        "Education among adults aged 20+: 1 = less than 9th grade; "
        "2 = 9-11th grade or 12th grade without diploma; "
        "3 = high school/GED; 4 = some college/associate degree; "
        "5 = college graduate or above. Special codes set missing."
    ),
    "DMDMARTZ": (
        "Marital status: 1 = married/living with partner; "
        "2 = widowed/divorced/separated; 3 = never married. "
        "Special codes set missing."
    ),
    "INDFMPIR": (
        "Family income-to-poverty ratio; 5.00 represents 5.00 or more."
    ),
    "DMDHHSIZ": (
        "Household size; 7 represents 7 or more people."
    ),
    "average_sleep_hours": (
        "Weighted average sleep: (5 * weekday hours + "
        "2 * weekend hours) / 7; computed only when both values "
        "are in the documented 2-14 range."
    ),
    "SMQ020": (
        "Ever smoked at least 100 cigarettes: 1 = yes, 0 = no; "
        "special/invalid values set missing."
    ),
    "SMQ040": (
        "Current smoking frequency among respondents routed to "
        "this question: 1 = every day, 2 = some days, "
        "3 = not at all; special/invalid values set missing."
    ),
    "ALQ121": (
        "Estimated annual number of drinking occasions derived "
        "from NHANES past-year frequency categories. Approximate, "
        "not an exact count."
    ),
    "ALQ130": (
        "Average drinks per drinking day: source values 1-14 retained; "
        "15 represents 15 or more; set to zero when ALQ121 indicates "
        "never drinking; missing when drinking frequency is invalid."
    ),
    "ALQ142": (
        "Estimated annual number of heavy drinking occasions, "
        "derived from NHANES frequency categories. Approximate, "
        "not an exact count."
    ),
    "moderate_activity_min_week": (
        "Estimated minutes per week of moderate-intensity "
        "leisure-time physical activity."
    ),
    "vigorous_activity_min_week": (
        "Estimated minutes per week of vigorous-intensity "
        "leisure-time physical activity."
    ),
    "sedentary_minutes": (
        "Minutes spent sitting on a typical day, from PAD680."
    ),
    "FSDAD": (
        "Adult food security category: 1 = full; 2 = marginal; "
        "3 = low; 4 = very low food security."
    ),
    "INQ300": (
        "Family has more than $5,000 in savings/cash assets: "
        "1 = yes, 0 = no; special/invalid values set missing."
    ),
    "moderate_activity_category": (
        "Alternative categorical version of moderate weekly activity: "
        "low, medium, high based on approximate thirds of valid values."
    ),
    "vigorous_activity_category": (
        "Alternative categorical version of vigorous weekly activity: "
        "low, medium, high based on approximate thirds of valid values."
    ),
}

dictionary = pd.DataFrame({
    "feature": final_columns,
    "description": [
        feature_descriptions[column] for column in final_columns
    ],
    "role": [
        "identifier" if column == "SEQN"
        else "target" if column == "PHQ9_total"
        else "predictor"
        for column in final_columns
    ],
})


# ============================================================
# 17. SAVE OUTPUTS
# ============================================================

dataset.to_csv(DATASET_FILE, index=False)
dictionary.to_csv(DICTIONARY_FILE, index=False)


# ============================================================
# 18. REPORT
# ============================================================

print("\n" + "=" * 65)
print("PREPARATION COMPLETED")
print("=" * 65)

print(f"Rows in final dataset: {len(dataset)}")
print(f"Columns in final dataset: {len(dataset.columns)}")
print(f"Dataset saved to: {DATASET_FILE}")
print(f"Feature dictionary saved to: {DICTIONARY_FILE}")

print("\nPHQ9_total statistics:")
print(dataset["PHQ9_total"].describe())

print("\nMissing values by column:")
print(dataset.isna().sum().to_string())

print("\nModerate activity category counts:")
print(
    dataset["moderate_activity_category"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nVigorous activity category counts:")
print(
    dataset["vigorous_activity_category"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nFirst five rows:")
print(dataset.head().to_string(index=False))

print("\nDone.")
