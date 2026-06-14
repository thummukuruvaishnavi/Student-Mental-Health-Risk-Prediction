import numpy as np
import pandas as pd

np.random.seed(42)
rows = 500

data = pd.DataFrame({
    "study_hours": np.random.uniform(1, 10, rows),
    "sleep_hours": np.random.uniform(4, 9, rows),
    "social_media": np.random.uniform(0, 6, rows),
    "exercise": np.random.randint(0, 6, rows),
    "pressure": np.random.randint(1, 11, rows),
    "attendance": np.random.uniform(50, 100, rows)
})

data["risk"] = (
    (data["sleep_hours"] < 6) |
    (data["social_media"] > 4) |
    (data["pressure"] > 7)
).astype(int)

data.to_csv("student_mental_health_500.csv", index=False)
print("Dataset created successfully!")
