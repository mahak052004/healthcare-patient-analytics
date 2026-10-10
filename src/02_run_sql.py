
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "outputs" / "healthcare_analytics.db"
SQL_FILE = ROOT / "sql" / "analysis_queries.sql"
OUT = ROOT / "outputs"

sql_text = SQL_FILE.read_text(encoding="utf-8")

# Run each semicolon-separated query individually.
queries = [
    q.strip()
    for q in sql_text.split(";")
    if q.strip()
]

with sqlite3.connect(DB) as conn:
    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, conn)

        print(f"\nQUERY {number}")
        print(result.to_string(index=False))

        result.to_csv(
            OUT / f"sql_result_{number:02d}.csv",
            index=False,
        )

print("\nAll SQL queries executed.")
