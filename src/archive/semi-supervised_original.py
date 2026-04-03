import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
import joblib

# ========= 1. Load data =========

LABELED_PATH = "processed_students.csv"
UNLABELED_PATH = "processed_students_raw.csv"
TARGET_COL = "risk_level"

EXCLUDED_COLS = [
    "student_id",
    "role",
    "username",
    "password",
    "educational_program",
]

df_labeled = pd.read_csv(LABELED_PATH)
df_unlabeled = pd.read_csv(UNLABELED_PATH)

print("Labeled shape:", df_labeled.shape)
print("Unlabeled shape:", df_unlabeled.shape)

# ========= 2. Split features/target and drop excluded columns =========

# Labeled
X_labeled = df_labeled.drop(columns=EXCLUDED_COLS + [TARGET_COL])
y_labeled = df_labeled[TARGET_COL]

# Unlabeled – same feature columns, but no target
X_unlabeled = df_unlabeled.drop(columns=EXCLUDED_COLS)

# Make sure columns match between labeled and unlabeled
assert list(X_labeled.columns) == list(X_unlabeled.columns), "Feature columns differ between labeled and unlabeled!"

feature_cols = X_labeled.columns.tolist()

# Treat everything as categorical for now
categorical_features = feature_cols

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="drop"
)

# ========= 3. Define base classifiers =========

base_models = {
    "log_reg": LogisticRegression(max_iter=1000, multi_class="auto"),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ),
    "grad_boost": GradientBoostingClassifier(
        random_state=42
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=400,
        random_state=42
    )
}

# ========= 4. Semi-supervised evaluation function =========

def evaluate_ssl_model(name, base_estimator, X_labeled, y_labeled, X_unlabeled,
                       n_splits=4, threshold=0.8, random_state=42):
    """
    For each fold:
      - Train on: labeled_train + all unlabeled (with y = -1 for unlabeled)
      - Validate on: labeled_val
    Returns mean accuracy & macro-F1.
    """
    print(f"\n=== Evaluating model: {name} ===")

    y_labeled = pd.Series(y_labeled).reset_index(drop=True)
    X_labeled_local = X_labeled.reset_index(drop=True)
    X_unlabeled_local = X_unlabeled.reset_index(drop=True)

    # Choose number of folds based on smallest class size
    class_counts = y_labeled.value_counts()
    min_class = class_counts.min()
    if n_splits > min_class:
        n_splits = max(2, min_class)
        print(f"Adjusted number of folds to {n_splits} due to small class sizes.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_scores = []
    f1_scores = []

    fold = 1
    for train_idx, val_idx in skf.split(X_labeled_local, y_labeled):
        print(f"\n--- Fold {fold} ---")
        fold += 1

        X_train_l = X_labeled_local.iloc[train_idx]
        y_train_l = y_labeled.iloc[train_idx]

        X_val = X_labeled_local.iloc[val_idx]
        y_val = y_labeled.iloc[val_idx]

        # Combine labeled train + all unlabeled for semi-supervised training
        X_train_full = pd.concat([X_train_l, X_unlabeled_local], axis=0).reset_index(drop=True)
        y_train_full = pd.concat([
            y_train_l,
            pd.Series([-1] * len(X_unlabeled_local))
        ], ignore_index=True)

        # Build pipeline with fresh clones
        base_clf = clone(base_estimator)
        ssl_clf = SelfTrainingClassifier(
            base_clf,
            criterion="threshold",
            threshold=threshold
        )

        model = Pipeline(steps=[
            ("preprocess", clone(preprocessor)),
            ("ssl", ssl_clf),
        ])

        # Fit on labeled_train + unlabeled
        model.fit(X_train_full, y_train_full)

        # Evaluate on validation labeled only
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        acc_scores.append(acc)
        f1_scores.append(f1)

        print("Fold accuracy:", acc)
        print("Fold macro-F1:", f1)

    mean_acc = float(np.mean(acc_scores))
    mean_f1 = float(np.mean(f1_scores))

    print(f"\n>>> {name} | Mean accuracy: {mean_acc:.4f}, Mean macro-F1: {mean_f1:.4f}")
    return {
        "name": name,
        "accuracy": mean_acc,
        "macro_f1": mean_f1,
    }

# ========= 5. Run evaluation for all models =========

results = []
for name, base_est in base_models.items():
    res = evaluate_ssl_model(
        name=name,
        base_estimator=base_est,
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        X_unlabeled=X_unlabeled,
        n_splits=4,
        threshold=0.8,
        random_state=42
    )
    results.append(res)

print("\n=== Summary of models ===")
for r in results:
    print(f"{r['name']:15s} | acc={r['accuracy']:.4f} | macro_f1={r['macro_f1']:.4f}")

# ========= 6. Choose best model by macro-F1 =========

best = max(results, key=lambda r: r["macro_f1"])
best_name = best["name"]
print(f"\n*** Best model: {best_name} (macro-F1={best['macro_f1']:.4f}) ***")

best_base_estimator = base_models[best_name]

# ========= 7. Train final model on ALL labeled + ALL unlabeled =========

print("\nTraining final semi-supervised model on full data...")

X_labeled_full = X_labeled.reset_index(drop=True)
y_labeled_full = y_labeled.reset_index(drop=True)
X_unlabeled_full = X_unlabeled.reset_index(drop=True)

X_train_all = pd.concat([X_labeled_full, X_unlabeled_full], axis=0).reset_index(drop=True)
y_train_all = pd.concat([
    y_labeled_full,
    pd.Series([-1] * len(X_unlabeled_full))
], ignore_index=True)

best_base_clf = clone(best_base_estimator)
best_ssl_clf = SelfTrainingClassifier(
    best_base_clf,
    criterion="threshold",
    threshold=0.8
)

final_model = Pipeline(steps=[
    ("preprocess", clone(preprocessor)),
    ("ssl", best_ssl_clf),
])

final_model.fit(X_train_all, y_train_all)

# OPTIONAL: Show performance on labeled data (transductive check)
y_pred_labeled = final_model.predict(X_labeled_full)
print("\n=== Performance on all labeled students (for sanity check) ===")
print("Accuracy:", accuracy_score(y_labeled_full, y_pred_labeled))
print("Macro-F1:", f1_score(y_labeled_full, y_pred_labeled, average="macro"))
print("\nClassification report:")
print(classification_report(y_labeled_full, y_pred_labeled))

# ========= 8. Save model =========

MODEL_PATH = "student_risk_semi_supervised.pkl"
joblib.dump(final_model, MODEL_PATH)
print(f"\nFinal model saved to: {MODEL_PATH}")
