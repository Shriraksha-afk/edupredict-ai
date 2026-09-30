from pathlib import Path
import numpy as np
import pandas as pd

# Synthetic data for demonstrating the pipeline. Not real student records.
rng = np.random.default_rng(42)
n = 1200
attendance = rng.uniform(35, 100, n)
study = rng.uniform(0, 8, n)
assignment = rng.uniform(20, 100, n)
internal1 = rng.uniform(15, 100, n)
internal2 = rng.uniform(15, 100, n)
participation = rng.integers(1, 6, n)
sleep = rng.uniform(4, 10, n)

noise = rng.normal(0, 8, n)
score = (
    0.20 * attendance + 0.12 * assignment +
    0.20 * internal1 + 0.22 * internal2 +
    2.2 * study + 1.2 * participation +
    0.8 * sleep + noise - 25
)
score = np.clip(score, 0, 100).round(1)

df = pd.DataFrame({
    "attendance_percent": attendance.round(1),
    "study_hours_per_day": study.round(1),
    "assignment_score": assignment.round(1),
    "internal_exam1": internal1.round(1),
    "internal_exam2": internal2.round(1),
    "class_participation": participation,
    "sleep_hours": sleep.round(1),
    "final_score": score
})
Path("data").mkdir(exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)
print(f"Created {len(df)} synthetic rows at data/student_performance.csv")
