import joblib
import pandas as pd
from preprocessing import preprocess_apply

# pipeline = joblib.load("pipeline.pkl")
model = joblib.load("student_risk_semi_supervised.pkl")   # example

new = pd.read_csv("new_student.csv")

pred = model.predict(new)

print(pred)
