
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. LOAD THE FIVE SOURCE FILES
# --------------------------------------------------
patients = pd.read_csv(DATA / "patients.csv")
admissions = pd.read_csv(DATA / "admissions.csv")
vitals = pd.read_csv(DATA / "vitals.csv")
treatments = pd.read_csv(DATA / "treatments.csv")
doctors = pd.read_csv(DATA / "doctors.csv")

tables = {
    "patients": patients,
    "admissions": admissions,
    "vitals": vitals,
    "treatments": treatments,
    "doctors": doctors,
}

# --------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES AND EMPTY STRINGS
# --------------------------------------------------
for name, df in tables.items():
    df.columns = df.columns.str.strip().str.lower()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        df[col] = df[col].replace("", np.nan)

# --------------------------------------------------
# 3. PARSE DATES
# --------------------------------------------------
patients["dob"] = pd.to_datetime(
    patients["dob"], errors="coerce"
)

for col in ["admission_date", "discharge_date"]:
    admissions[col] = pd.to_datetime(
        admissions[col], errors="coerce"
    )

vitals["recorded_time"] = pd.to_datetime(
    vitals["recorded_time"], errors="coerce"
)

treatments["treatment_date"] = pd.to_datetime(
    treatments["treatment_date"], errors="coerce"
)

# --------------------------------------------------
# 4. CHECK PRIMARY KEYS
# --------------------------------------------------
primary_keys = {
    "patients": ("patient_id", patients),
    "admissions": ("admission_id", admissions),
    "vitals": ("vital_id", vitals),
    "treatments": ("treatment_id", treatments),
    "doctors": ("doctor_id", doctors),
}

for name, (key, df) in primary_keys.items():
    if df[key].isna().any():
        raise ValueError(f"{name}: missing primary key {key}")
    if df[key].duplicated().any():
        raise ValueError(f"{name}: duplicate primary key {key}")

# --------------------------------------------------
# 5. CHECK FOREIGN KEYS
# --------------------------------------------------
checks = {
    "admissions_patient": (
        admissions["patient_id"], patients["patient_id"]
    ),
    "admissions_doctor": (
        admissions["doctor_id"], doctors["doctor_id"]
    ),
    "vitals_admission": (
        vitals["admission_id"], admissions["admission_id"]
    ),
    "treatments_admission": (
        treatments["admission_id"], admissions["admission_id"]
    ),
}

for name, (child, parent) in checks.items():
    orphan_count = (~child.isin(parent)).sum()
    print(f"{name}: {orphan_count} unmatched foreign keys")

# Do not silently delete records if orphans are found.
# Review them before deciding how to handle them.

# --------------------------------------------------
# 6. CHECK ADMISSION DATES
# --------------------------------------------------
invalid_dates = (
    admissions["admission_date"].isna()
    | admissions["discharge_date"].isna()
    | (
        admissions["discharge_date"]
        < admissions["admission_date"]
    )
)

print("Invalid admission date rows:", int(invalid_dates.sum()))

# For this case study, stop rather than produce misleading
# length-of-stay calculations if admission dates are invalid.
if invalid_dates.any():
    raise ValueError(
        "Invalid admission dates found. Review source data."
    )

admissions["length_of_stay_days"] = (
    admissions["discharge_date"]
    - admissions["admission_date"]
).dt.days

# --------------------------------------------------
# 7. STANDARDIZE MISSING CHRONIC CONDITIONS
# --------------------------------------------------
patients["chronic_conditions"] = (
    patients["chronic_conditions"]
    .fillna("Unknown")
)

# "Unknown" means the value is missing, not that the
# patient definitely has no chronic conditions.

# --------------------------------------------------
# 8. CHECK VITALS AND TREATMENT TIMING
# --------------------------------------------------
vital_check = vitals.merge(
    admissions[
        ["admission_id", "admission_date", "discharge_date"]
    ],
    on="admission_id",
    how="left",
    validate="many_to_one",
)

vital_check["recorded_date"] = (
    vital_check["recorded_time"].dt.normalize()
)

vital_check["outside_admission_window"] = (
    (vital_check["recorded_date"]
     < vital_check["admission_date"])
    | (vital_check["recorded_date"]
       > vital_check["discharge_date"])
)

treatment_check = treatments.merge(
    admissions[
        ["admission_id", "admission_date", "discharge_date"]
    ],
    on="admission_id",
    how="left",
    validate="many_to_one",
)

treatment_check["outside_admission_window"] = (
    (treatment_check["treatment_date"]
     < treatment_check["admission_date"])
    | (treatment_check["treatment_date"]
       > treatment_check["discharge_date"])
)

print(
    "Vital records outside admission window:",
    int(vital_check["outside_admission_window"].sum()),
    "of", len(vital_check)
)

print(
    "Treatments outside admission window:",
    int(treatment_check["outside_admission_window"].sum()),
    "of", len(treatment_check)
)

# Retain original rows for investigation. Do not delete
# these records without confirmation from the data owner.

# --------------------------------------------------
# 9. CREATE ADMISSION-LEVEL VITAL SUMMARY
# --------------------------------------------------
# These are descriptive summaries of linked records.
# Timing anomalies mean they should not be interpreted
# as confirmed in-hospital clinical measurements.

vital_summary = (
    vitals.groupby("admission_id")
    .agg(
        vital_record_count=("vital_id", "count"),
        avg_heart_rate=("heart_rate", "mean"),
        avg_oxygen_level=("oxygen_level", "mean"),
        avg_temperature=("temperature", "mean"),
    )
    .reset_index()
)

# --------------------------------------------------
# 10. CREATE TREATMENT SUMMARY
# --------------------------------------------------
treatment_summary = (
    treatments.groupby("admission_id")
    .agg(
        treatment_count=("treatment_id", "count"),
        distinct_medications=("medication", "nunique"),
        distinct_procedures=("procedure", "nunique"),
    )
    .reset_index()
)

# --------------------------------------------------
# 11. JOIN TABLES WITHOUT MULTIPLYING ROWS
# --------------------------------------------------
# Aggregate one-to-many tables before joining them.
# Joining raw vitals and treatments together can multiply
# the number of rows and inflate counts.

admission_features = (
    admissions
    .merge(
        patients,
        on="patient_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_patient"),
    )
    .merge(
        doctors,
        on="doctor_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_doctor"),
    )
    .merge(
        vital_summary,
        on="admission_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        treatment_summary,
        on="admission_id",
        how="left",
        validate="one_to_one",
    )
)

for col in [
    "vital_record_count",
    "treatment_count",
    "distinct_medications",
    "distinct_procedures",
]:
    admission_features[col] = (
        admission_features[col].fillna(0)
    )

# Age at admission is preferable to current age.
admission_features["age_at_admission"] = (
    admission_features["admission_date"]
    - admission_features["dob"]
).dt.days / 365.2425

# Negative or implausible ages need review.
admission_features.loc[
    (admission_features["age_at_admission"] < 0)
    | (admission_features["age_at_admission"] > 120),
    "age_at_admission"
] = np.nan

# --------------------------------------------------
# 12. SAVE CLEANED TABLES AND REPORTING DATASETS
# --------------------------------------------------
for name, df in tables.items():
    df.to_csv(OUT / f"clean_{name}.csv", index=False)

admission_features.to_csv(
    OUT / "admission_features.csv", index=False
)

vital_check.to_csv(
    OUT / "vital_quality_audit.csv", index=False
)

treatment_check.to_csv(
    OUT / "treatment_quality_audit.csv", index=False
)

# --------------------------------------------------
# 13. DATA QUALITY REPORT
# --------------------------------------------------
quality_rows = []

for name, df in tables.items():
    quality_rows.append({
        "table_name": name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    })

quality = pd.DataFrame(quality_rows)
quality.to_csv(OUT / "data_quality_report.csv", index=False)

# --------------------------------------------------
# 14. CREATE A LOCAL SQLITE DATABASE
# --------------------------------------------------
# SQLite stores the tables locally so we can query them
# using SQL without configuring a cloud database.

db_path = OUT / "healthcare_analytics.db"

with sqlite3.connect(db_path) as conn:
    for name, df in tables.items():
        df.to_sql(
            name,
            conn,
            if_exists="replace",
            index=False,
        )

    admission_features.to_sql(
        "admission_features",
        conn,
        if_exists="replace",
        index=False,
    )

print("\nETL completed successfully.")
print("Data quality report:", OUT / "data_quality_report.csv")
print("Admission features:", OUT / "admission_features.csv")
print("Local SQL database:", db_path)
print("\nQuality report:")
print(quality.to_string(index=False))
