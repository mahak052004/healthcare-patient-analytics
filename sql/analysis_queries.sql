
-- 1. Total admissions and unique patients
SELECT
    COUNT(*) AS total_admissions,
    COUNT(DISTINCT patient_id) AS unique_patients
FROM admissions;


-- 2. Monthly admission trends
SELECT
    strftime('%Y-%m', admission_date) AS admission_month,
    COUNT(*) AS total_admissions
FROM admissions
GROUP BY strftime('%Y-%m', admission_date)
ORDER BY admission_month;


-- 3. Average length of stay by diagnosis
SELECT
    diagnosis,
    COUNT(*) AS admission_count,
    ROUND(AVG(length_of_stay_days), 2) AS avg_stay_days
FROM admission_features
GROUP BY diagnosis
ORDER BY admission_count DESC;


-- 4. Most common diagnoses
SELECT
    diagnosis,
    COUNT(*) AS admission_count
FROM admissions
GROUP BY diagnosis
ORDER BY admission_count DESC
LIMIT 10;


-- 5. Patients with repeated admissions
SELECT
    patient_id,
    COUNT(*) AS admission_count
FROM admissions
GROUP BY patient_id
HAVING COUNT(*) > 1
ORDER BY admission_count DESC;


-- 6. Admissions by doctor and specialization
SELECT
    doctor_id,
    specialization,
    COUNT(*) AS admission_count
FROM admission_features
GROUP BY doctor_id, specialization
ORDER BY admission_count DESC;


-- 7. Treatment counts by admission
SELECT
    admission_id,
    treatment_count,
    distinct_medications,
    distinct_procedures
FROM admission_features
ORDER BY treatment_count DESC
LIMIT 20;


-- 8. Admissions with missing chronic-condition information
SELECT
    chronic_conditions,
    COUNT(*) AS admission_count
FROM admission_features
GROUP BY chronic_conditions
ORDER BY admission_count DESC;
