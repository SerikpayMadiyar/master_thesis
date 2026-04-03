import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------------------------------------------------------
# Helper converters
# --------------------------------------------------------

def gpa_to_num(g):
    if pd.isna(g):
        return np.nan
    if "below" in g:
        return 2.6
    if "2.8-3.0" in g:
        return 2.9
    if "3.1-3.2" in g:
        return 3.15
    if "3.3-3.4" in g:
        return 3.35
    if "above" in g:
        return 3.5
    return np.nan

def attendance_to_num(a):
    if pd.isna(a):
        return np.nan
    a = a.replace("%", "")
    if "below" in a:
        return 69
    if "70-79" in a:
        return 74.5
    if "80-89" in a:
        return 84.5
    if "90-100" in a:
        return 95
    return np.nan

# --------------------------------------------------------
# Preprocessing function used for TRAINING
# --------------------------------------------------------

def preprocess_fit(df):
    df = df.copy()

    # lowercase everything
    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # drop technical columns
    drop_cols = ["student_id", "role", "username", "password"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # drop duplicates
    df = df.drop_duplicates()

    # numerical conversions
    df["avg_gpa_num"] = df["average_gpa"].apply(gpa_to_num)
    df["attendance_num"] = df["average_attendance"].apply(attendance_to_num)

    study_map = {
        "do not prepare": 0,
        "less than 1 hour": 1,
        "1-3 hours": 2,
        "more than 4 hours": 3
    }
    df["study_num"] = df["self_study_hours"].map(study_map)

    sleep_map = {
        "less than 6 hours": 5,
        "7-9 hours": 8,
        "more than 10 hours": 11
    }
    df["sleep_num"] = df["average_sleep_hours"].map(sleep_map)

    prep_map = {
        "less than 1 hour": 0,
        "2–8 hours": 5,
        "9–15 hours": 12,
        "more than 16 hours": 16
    }
    df["prep_num"] = df["exam_preparation_hours"].map(prep_map)

    # extract labels
    y = df["risk_level"]
    df = df.drop(columns=[
        "risk_level","average_gpa","average_attendance",
        "self_study_hours","average_sleep_hours","exam_preparation_hours"
    ])

    # column categories
    nominal_cols = [
        "gender","degree","educational_program","preferred_field_of_study",
        "has_job","financial_support_from_family","most_influencing_factor"
    ]

    ordinal_cols = {
        "age": ["16-17","18-19","20-21","22-23","above 24"],
        "year_of_study": ["1 year","2 year","3 year"],
        "motivation_level": ["very low","low","average","high","very high"],
        "stress_frequency": ["never","rarely","sometimes","often","very often"],
        "importance_of_interest_in_subject": [
            "not important","slightly important","moderately important",
            "important","very important"
        ],
        "communication_skills": ["very low","low","average","high","very high"],
        "teacher_competence_satisfaction": [
            "not satisfied at all","rather dissatisfied","hard to say",
            "rather satisfied","fully satisfied"
        ],
        "use_of_university_resources": [
            "never","rarely","sometimes","often","very often"
        ],
        "risk_level": [
            "low","medium","high"
        ]
    }

    ordinal_feature_list = list(ordinal_cols.keys())

    numeric_cols = ["avg_gpa_num","attendance_num","study_num",
                    "sleep_num","prep_num"]

    # BUILD PIPELINE
    preprocess = ColumnTransformer(
        transformers=[
            ("nominal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))
            ]),
            nominal_cols),

            ("ordinal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(categories=[ordinal_cols[c] for c in ordinal_cols]))
            ]),
            ordinal_feature_list),

            ("num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler())
            ]),
            numeric_cols)
        ]
    )


    pipeline = Pipeline([
        ("preprocess", preprocess)
    ])

    X = pipeline.fit_transform(df)
    print("Preprocessing complete.")
    print("X shape:", X.shape)
    print(df.head())

    return X, y, pipeline


# --------------------------------------------------------
# Preprocessing function used for TESTING / PREDICTION
# --------------------------------------------------------

def preprocess_apply(df, pipeline):
    df = df.copy()

    # Same cleaning rules MUST be applied
    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # drop technical columns
    drop_cols = ["student_id", "role", "username", "password"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # numerical conversions
    df["avg_gpa_num"] = df["average_gpa"].apply(gpa_to_num)
    df["attendance_num"] = df["average_attendance"].apply(attendance_to_num)

    study_map = {
        "do not prepare": 0,
        "less than 1 hour": 1,
        "1-3 hours": 2,
        "more than 4 hours": 3
    }
    df["study_num"] = df["self_study_hours"].map(study_map)

    sleep_map = {
        "less than 6 hours": 5,
        "7-9 hours": 8,
        "more than 10 hours": 11
    }
    df["sleep_num"] = df["average_sleep_hours"].map(sleep_map)

    prep_map = {
        "less than 1 hour": 0,
        "2–8 hours": 5,
        "9–15 hours": 12,
        "more than 16 hours": 16
    }
    df["prep_num"] = df["exam_preparation_hours"].map(prep_map)

    # drop unused
    df = df.drop(columns=[
        "average_gpa","average_attendance","self_study_hours",
        "average_sleep_hours","exam_preparation_hours"
    ])

    X = pipeline.transform(df)

    return X
