import joblib
import pandas as pd
from preprocess import preprocess_apply

MODEL_PATH = "./student_risk_semi_supervised.pkl"
PREPROCESS_PATH = "./preprocess_pipeline.pkl"

model = joblib.load(MODEL_PATH)
preprocess_pipeline = joblib.load(PREPROCESS_PATH)

new = pd.read_csv("/home/madiyar/work/homelab/master-thesis/data/new_student.csv")

X_new = preprocess_apply(new, preprocess_pipeline)
pred = model.predict(X_new)

print(pred)
