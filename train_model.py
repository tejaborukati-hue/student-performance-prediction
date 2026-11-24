
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project folder
BASE_DIR = Path(__file__).parent

# Generate sample student data
np.random.seed(42)

data = {
    "study_hours": np.random.randint(1, 10, 500),
    "attendance": np.random.randint(40, 101, 500),
    "previous_marks": np.random.randint(30, 101, 500),
    "assignments": np.random.randint(1, 11, 500)
}

df = pd.DataFrame(data)

# Calculate sample performance score
score = (
    df["study_hours"] * 4 +
    df["attendance"] * 0.3 +
    df["previous_marks"] * 0.4 +
    df["assignments"] * 2
)

# Create result labels
df["result"] = np.where(score >= 65, 1, 0)

# Save dataset
df.to_csv(BASE_DIR / "student_data.csv", index=False)

# Separate input and output
X = df[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignments"
    ]
]

y = df["result"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Model Accuracy:", accuracy * 100, "%")

print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, BASE_DIR / "student_model.pkl")

print("Model saved successfully!")
