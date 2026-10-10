# Healthcare Patient Analytics & Prediction

## Project Overview

This project develops a healthcare analytics workflow for analyzing hospital admissions, patient demographics, treatment records, and recorded vital signs. It demonstrates data cleaning, SQL analytics, retrospective 30-day readmission prediction, and interactive Power BI reporting.

## Business Objectives

- Analyze admission and discharge patterns.
- Identify patterns associated with historical readmissions.
- Summarize recorded vital signs and investigate data-quality issues.
- Develop an analytical framework for post-discharge monitoring.
- Support data-driven operational reporting.

## Technology Stack

- Python
- Pandas and NumPy
- SQLite and SQL
- Scikit-learn
- Power BI Desktop
- Git and GitHub

## Project Structure

- `src/`: Python ETL, SQL execution, and model training scripts.
- `sql/`: SQL analytics queries.
- `data/`: Local source CSV files; excluded from public version control.
- `outputs/`: Locally generated data, database, and evaluation results.
- `models/`: Locally saved model artifact.
- `screenshots/`: Actual project and dashboard screenshots.

## Implementation

1. Profile the five source datasets.
2. Validate keys, dates, and table relationships.
3. Create admission-level analytical features.
4. Generate a local SQLite database.
5. Execute SQL reporting queries.
6. Derive an observable 30-day readmission target.
7. Train and evaluate a Random Forest classifier.
8. Build five Power BI report pages.
9. Document findings and limitations.

## Run the Project

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Place the authorized source CSV files in `data/`, then run:

```bash
python src/01_etl.py
python src/02_run_sql.py
python src/03_train_model.py
```

## Important Limitations

The supplied data contains inconsistencies between some recorded vital/treatment timestamps and their linked admission windows. These records require source-system investigation. The readmission target is derived from recorded admission history, and model results are exploratory rather than clinically validated.

The follow-up dashboard is a monitoring prototype because appointment-level records were not supplied.

No patient-level data should be published in this repository.

## Deliverables

- Python source code
- SQL analysis queries
- Power BI report file maintained locally
- Actual dashboard screenshots
- Project documentation and findings