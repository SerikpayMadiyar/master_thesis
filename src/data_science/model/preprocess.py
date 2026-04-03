import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def gpa_to_num(g):
    if pd.isna(g):
        return np.nan
    g = g.lower()
    if "below" in g:
        return 2.6
    if "2.8-3.0" in g:
        return 2.9
    if "3.1-3.2" in g:
        return 3.15
    if "3.3-3.4" in g:
        return 3.35
    if "above" in g:
        return 3.6
    return np.nan


def attendance_to_num(a):
    if pd.isna(a):
        return np.nan
    a = a.lower().replace("%", "")
    if "below" in a:
        return 65
    if "70-79" in a:
        return 74.5
    if "80-89" in a:
        return 84.5
    if "90-100" in a:
        return 95
    return np.nan


def study_to_num(s):
    if pd.isna(s):
        return np.nan
    s = s.lower()
    if "do not" in s:
        return 0
    if "less" in s:
        return 1
    if "1-3" in s:
        return 2
    if "more than 4" in s:
        return 3
    return np.nan


def sleep_to_num(s):
    if pd.isna(s):
        return np.nan
    s = s.lower()
    if "less" in s:
        return 5
    if "7-9" in s:
        return 8
    if "more" in s:
        return 11
    return np.nan


def prep_to_num(s):
    if pd.isna(s):
        return np.nan
    s = s.lower()
    if "less" in s:
        return 0
    if "2–8" in s or "2-8" in s:
        return 5
    if "9–15" in s or "9-15" in s:
        return 12
    if "more" in s:
        return 16
    return np.nan


def preprocess_fit(df):

    df = df.copy()

    # Standardize everything
    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Remove unusable columns
    drop_cols = ["student_id", "role", "username", "password", "educational_program"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove duplicates
    df = df.drop_duplicates()

    # Extract label
    y = None
    if "risk_level" in df.columns:
        y = df["risk_level"].copy()
        df = df.drop(columns=["risk_level"])

    # Numerical conversions
    df["gpa_num"] = df["average_gpa"].apply(gpa_to_num)
    df["attend_num"] = df["average_attendance"].apply(attendance_to_num)
    df["study_num"] = df["self_study_hours"].apply(study_to_num)
    df["sleep_num"] = df["average_sleep_hours"].apply(sleep_to_num)
    df["prep_num"] = df["exam_preparation_hours"].apply(prep_to_num)

    # Drop original range columns
    df = df.drop(columns=[
        "average_gpa", "average_attendance",
        "self_study_hours", "average_sleep_hours", "exam_preparation_hours"
    ])

    # Feature groups
    nominal_cols = [
        "gender", "degree", "preferred_field_of_study",
        "has_job", "financial_support_from_family",
        "most_influencing_factor"
    ]

    # Ordinal categories CORRECTED + EXPANDED
    ordinal_mapping = {
        "age": ["16-17", "18-19", "20-21", "22-23", "24-25", "above 24"],
        "year_of_study": ["1 year", "2 year", "3 year", "4 year"],  # flexible
        "motivation_level": ["very low", "low", "average", "high", "very high"],
        "stress_frequency": ["never", "rarely", "sometimes", "often", "very often"],
        "importance_of_interest_in_subject": [
            "not important", "slightly important", "moderately important",
            "important", "very important"
        ],
        "communication_skills": ["very low", "low", "average", "high", "very high"],
        "teacher_competence_satisfaction": [
            "not satisfied at all", "rather dissatisfied", "hard to say",
            "rather satisfied", "fully satisfied"
        ],
        "use_of_university_resources": [
            "never", "rarely", "sometimes", "often", "very often"
        ]
    }

    ordinal_cols = list(ordinal_mapping.keys())

    numeric_cols = ["gpa_num", "attend_num", "study_num", "sleep_num", "prep_num"]

    # ColumnTransformer pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("nominal",
             Pipeline([
                 ("impute", SimpleImputer(strategy="most_frequent")),
                 ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
             ]),
             nominal_cols),

            ("ordinal",
             Pipeline([
                 ("impute", SimpleImputer(strategy="most_frequent")),
                 ("encode", OrdinalEncoder(categories=[ordinal_mapping[c] for c in ordinal_cols]))
             ]),
             ordinal_cols),

            ("numeric",
             Pipeline([
                 ("impute", SimpleImputer(strategy="median")),
                 ("scale", MinMaxScaler())
             ]),
             numeric_cols)
        ],
        remainder="drop"
    )

    # Full pipeline
    pipeline = Pipeline([
        ("prep", preprocess)
    ])

    X = pipeline.fit_transform(df)

    print("✔ Preprocessing fit complete.")
    print("Final feature matrix shape:", X.shape)

    return X, y, pipeline


def preprocess_apply(df, pipeline):

    df = df.copy()

    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    drop_cols = ["student_id", "role", "username", "password", "educational_program"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # apply same numeric conversions
    df["gpa_num"] = df["average_gpa"].apply(gpa_to_num)
    df["attend_num"] = df["average_attendance"].apply(attendance_to_num)
    df["study_num"] = df["self_study_hours"].apply(study_to_num)
    df["sleep_num"] = df["average_sleep_hours"].apply(sleep_to_num)
    df["prep_num"] = df["exam_preparation_hours"].apply(prep_to_num)

    df = df.drop(columns=[
        "average_gpa", "average_attendance",
        "self_study_hours", "average_sleep_hours", "exam_preparation_hours"
    ])

    X = pipeline.transform(df)

    return X
