import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# -----------------------------------------
# 1. Load your data
# -----------------------------------------
df = pd.read_csv("processed_students_raw.csv")

# -----------------------------------------
# 2. Preprocess (FIT pipeline)
# -----------------------------------------
from preprocessing import preprocess_fit
X, y, pipeline = preprocess_fit(df)

# save preprocessing pipeline
joblib.dump(pipeline, "pipeline.pkl")

# -----------------------------------------
# 3. Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# HELPER: Evaluate and print metrics
# =========================================
def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average="macro"))
    print("Recall:", recall_score(y_test, pred, average="macro"))
    print("F1:", f1_score(y_test, pred, average="macro"))
    print("\nClassification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# =========================================
# 4. MODELS + HYPERPARAMETER TUNING
# =========================================

# -----------------------------
# 1) SVM
# -----------------------------
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm = GridSearchCV(SVC(), svm_params, cv=3)
svm.fit(X_train, y_train)

joblib.dump(svm.best_estimator_, "model_svm.pkl")
evaluate(svm.best_estimator_, X_test, y_test, "SVM")


# -----------------------------
# 2) Logistic Regression
# -----------------------------
log_params = {
    "C": [0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}

log_reg = GridSearchCV(LogisticRegression(max_iter=500), log_params, cv=3)
log_reg.fit(X_train, y_train)

joblib.dump(log_reg.best_estimator_, "model_logreg.pkl")
evaluate(log_reg.best_estimator_, X_test, y_test, "Logistic Regression")


# -----------------------------
# 3) Random Forest
# -----------------------------
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5],
}

rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=3)
rf.fit(X_train, y_train)

joblib.dump(rf.best_estimator_, "model_rf.pkl")
evaluate(rf.best_estimator_, X_test, y_test, "Random Forest")


# -----------------------------
# 4) KNN
# -----------------------------
knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3)
knn.fit(X_train, y_train)

joblib.dump(knn.best_estimator_, "model_knn.pkl")
evaluate(knn.best_estimator_, X_test, y_test, "KNN")


# -----------------------------
# 5) Naive Bayes
# -----------------------------
# GaussianNB has no hyperparameters, so we train directly.
nb = GaussianNB()
nb.fit(X_train.toarray() if hasattr(X_train, "toarray") else X_train, y_train)

joblib.dump(nb, "model_nb.pkl")
evaluate(nb, X_test.toarray() if hasattr(X_test, "toarray") else X_test, y_test, "Naive Bayes")


# -----------------------------
# 6) Decision Tree
# -----------------------------
dt_params = {
    "max_depth": [3, 5, 10, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5, 10]
}

dt = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=3)
dt.fit(X_train, y_train)

joblib.dump(dt.best_estimator_, "model_dt.pkl")
evaluate(dt.best_estimator_, X_test, y_test, "Decision Tree")
