
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. LOAD ADMISSION-LEVEL FEATURES
# --------------------------------------------------
df = pd.read_csv(
    OUT / "admission_features.csv",
    parse_dates=["admission_date", "discharge_date", "dob"],
)

df = df.sort_values(
    ["patient_id", "admission_date", "admission_id"]
).reset_index(drop=True)

# --------------------------------------------------
# 2. DERIVE THE NEXT ADMISSION DATE
# --------------------------------------------------
df["next_admission_date"] = (
    df.groupby("patient_id")["admission_date"].shift(-1)
)

df["days_to_next_admission"] = (
    df["next_admission_date"] - df["discharge_date"]
).dt.days

# --------------------------------------------------
# 3. DEFINE THE OBSERVATION CUTOFF
# --------------------------------------------------
latest_admission_date = df["admission_date"].max()
followup_cutoff = latest_admission_date - pd.Timedelta(days=30)

# An eligible admission must have a complete 30-day
# observation window in the available dataset.
eligible = df[
    df["discharge_date"] <= followup_cutoff
].copy()

# A readmission must occur after discharge and within
# 30 days. Same-day readmissions are included.
eligible["readmitted_30d"] = (
    eligible["days_to_next_admission"].between(0, 30)
).astype(int)

if eligible.empty:
    raise ValueError(
        "No admissions have a complete 30-day follow-up window."
    )

if eligible["readmitted_30d"].nunique() < 2:
    raise ValueError(
        "The eligible data contains only one target class. "
        "A binary classifier cannot be trained reliably."
    )

print("Latest admission date:", latest_admission_date.date())
print("Follow-up cutoff:", followup_cutoff.date())
print("Eligible admissions:", len(eligible))
print("Target distribution:")
print(eligible["readmitted_30d"].value_counts())

# --------------------------------------------------
# 4. CREATE PREDICTION FEATURES
# --------------------------------------------------
# Exclude patient names, phone numbers, address,
# IDs, next-admission date and target-related fields.
#
# We also exclude vital summaries because the timing
# audit found substantial inconsistencies.

numeric_features = [
    "age_at_admission",
    "length_of_stay_days",
    "treatment_count",
    "distinct_medications",
    "distinct_procedures",
]

categorical_features = [
    "gender",
    "chronic_conditions",
    "diagnosis",
    "specialization",
]

feature_columns = numeric_features + categorical_features

X = eligible[feature_columns].copy()
y = eligible["readmitted_30d"].copy()
groups = eligible["patient_id"].copy()

# --------------------------------------------------
# 5. PREPROCESS FEATURES
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 6. SPLIT BY PATIENT
# --------------------------------------------------
# Keeps a patient's admissions in one partition to
# reduce overly optimistic evaluation from patient overlap.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "The patient-grouped split does not contain both classes "
        "in train and test. More data or a different evaluation "
        "strategy is needed."
    )

# --------------------------------------------------
# 7. TRAIN A RANDOM FOREST MODEL
# --------------------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

# --------------------------------------------------
# 8. EVALUATE ON HELD-OUT PATIENTS
# --------------------------------------------------
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "eligible_admissions": int(len(eligible)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "positive_rate": float(y.mean()),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),
}

print("\nEvaluation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 9. SAVE EVALUATION OUTPUTS
# --------------------------------------------------
with open(OUT / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    OUT / "classification_report.csv"
)

evaluation = eligible.iloc[test_idx][
    ["admission_id", "patient_id", "admission_date",
     "discharge_date", "readmitted_30d"]
].copy()

evaluation["predicted_readmission"] = predictions
evaluation["predicted_probability"] = probabilities

evaluation.to_csv(
    OUT / "model_test_predictions.csv", index=False
)

# --------------------------------------------------
# 10. SAVE THE MODEL
# --------------------------------------------------
joblib.dump(model, MODEL_DIR / "readmission_model.joblib")

print("\nModel saved to models/readmission_model.joblib")
print("Metrics saved to outputs/model_metrics.json")
print("Test predictions saved to outputs/model_test_predictions.csv")
