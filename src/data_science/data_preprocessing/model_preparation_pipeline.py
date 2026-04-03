from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (BASE_DIR / "../data/output/processed_students_unlabeled.csv").resolve()

OUTPUT_DIR = (BASE_DIR / "../data/output/model_preparation").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_OUTPUT_FILE = OUTPUT_DIR / "students_harmonized_labeled.csv"
SCENARIO_A_FILE = OUTPUT_DIR / "students_scenario_a.csv"
SCENARIO_B_FILE = OUTPUT_DIR / "students_scenario_b.csv"
FEATURE_MANIFEST_FILE = OUTPUT_DIR / "feature_manifest.csv"
HARMONIZATION_REPORT_FILE = OUTPUT_DIR / "harmonization_report.csv"
UNEXPECTED_VALUES_REPORT_FILE = OUTPUT_DIR / "unexpected_values_after_harmonization.txt"
PREPARATION_SUMMARY_FILE = OUTPUT_DIR / "model_preparation_summary.txt"


# ============================================================
# Column groups
# ============================================================

TECHNICAL_COLUMNS = ["student_id", "role", "username", "password"]

TARGET_COLUMN = "academic_risk"

SURVEY_FEATURE_COLUMNS = [
    "gender",
    "age",
    "degree",
    "educational_program",
    "year_of_study",
    "average_gpa",
    "average_attendance",
    "preferred_field_of_study",
    "self_study_hours",
    "average_sleep_hours",
    "motivation_level",
    "has_job",
    "financial_support_from_family",
    "stress_frequency",
    "communication_skills",
    "teacher_competence_satisfaction",
    "use_of_university_resources",
    "importance_of_interest_in_subject",
    "exam_preparation_hours",
    "most_influencing_factor",
]

SCENARIO_A_FEATURES = SURVEY_FEATURE_COLUMNS.copy()

SCENARIO_B_FEATURES = [
    col
    for col in SURVEY_FEATURE_COLUMNS
    if col not in {"average_gpa", "average_attendance"}
]

NOMINAL_FEATURES = [
    "gender",
    "educational_program",
    "preferred_field_of_study",
    "most_influencing_factor",
]

ORDINAL_FEATURES = [
    "age",
    "degree",
    "year_of_study",
    "average_gpa",
    "average_attendance",
    "self_study_hours",
    "average_sleep_hours",
    "motivation_level",
    "has_job",
    "financial_support_from_family",
    "stress_frequency",
    "communication_skills",
    "teacher_competence_satisfaction",
    "use_of_university_resources",
    "importance_of_interest_in_subject",
    "exam_preparation_hours",
]


# ============================================================
# Canonical allowed values after harmonization
# ============================================================

CANONICAL_VALUES: dict[str, set[str]] = {
    "gender": {"male", "female"},
    "age": {"16-17", "18-19", "20-21", "22-23", "above 24"},
    "degree": {"bachelor", "master", "phd"},
    "year_of_study": {"1 year", "2 year", "3 year"},
    "average_gpa": {"below 2.7", "2.8-3.0", "3.1-3.2", "3.3-3.4", "above 3.5"},
    "average_attendance": {"below 70%", "70-79%", "80-89%", "90-100%"},
    "preferred_field_of_study": {
        "humanities",
        "technical sciences",
        "computer sciences",
    },
    "self_study_hours": {
        "do not study",
        "less than 1 hour",
        "1-3 hours",
        "more than 4 hours",
    },
    "average_sleep_hours": {"less than 6 hours", "7-9 hours", "more than 10 hours"},
    "motivation_level": {"very high", "high", "average", "low", "very low"},
    "has_job": {"yes, full-time", "yes, part-time", "no"},
    "financial_support_from_family": {
        "full support",
        "partial support",
        "no financial support",
    },
    "stress_frequency": {"very often", "often", "sometimes", "rarely", "never"},
    "communication_skills": {"very high", "high", "average", "low", "very low"},
    "teacher_competence_satisfaction": {
        "fully satisfied",
        "rather satisfied",
        "hard to say",
        "rather dissatisfied",
        "not satisfied at all",
    },
    "use_of_university_resources": {
        "very often",
        "often",
        "sometimes",
        "rarely",
        "never",
    },
    "importance_of_interest_in_subject": {
        "very important",
        "important",
        "moderately important",
        "slightly important",
        "not important",
    },
    "exam_preparation_hours": {
        "do not prepare",
        "less than 1 hour",
        "2-8 hours",
        "9-15 hours",
        "more than 16 hours",
    },
    "most_influencing_factor": {
        "personal motivation",
        "family support",
        "financial situation",
        "quality of teaching",
        "interest in the subject",
        "amount of self-study",
        "stress level",
    },
}

CANONICAL_EDUCATIONAL_PROGRAMS = {
    "6b06101 computer science",
    "6b06102 software engineering",
    "6b06103 big data analysis",
    "6b06105 media technologies",
    "6b06106 mathematical and computational science",
    "6b06801 big data in healthcare",
    "6b06301 cybersecurity",
    "6b06302 smart security technologies",
    "6b06202 smart technologies",
    "6b07101 industrial internet of things",
    "6b07102 electronic engineering",
    "6b07103 digital technologies in nuclear power engineering",
    "6b04101 it management",
    "6b04102 it entrepreneurship",
    "6b04103 ai business",
    "6b04104 digital public administration",
    "6b03201 digital journalism",
    "7m06103 applied data analytics",
    "7m06104 computational science",
    "7m06105 computer science and engineering",
    "7m06106 secure software engineering",
    "7m06107 media technologies",
    "7m06108 applied artificial intelligence",
    "8d06101 computer science",
    "8d06102 artificial intelligence",
    "8d04101 project management",
}


# ============================================================
# Harmonization rules
# ============================================================

HARMONIZATION_RULES: dict[str, dict[str, str]] = {
    "average_gpa": {
        "2.0-2.7": "below 2.7",
        "2.8-3.4": "3.1-3.2",
    },
    "average_attendance": {
        "50-69%": "below 70%",
    },
    "self_study_hours": {
        "do not prepare": "do not study",
    },
    "exam_preparation_hours": {
        "less than 5 hours": "2-8 hours",
        "5-10 hours": "2-8 hours",
        "11-20 hours": "9-15 hours",
        "more than 20 hours": "more than 16 hours",
    },
}


# ============================================================
# Utility functions
# ============================================================


def normalize_text(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip().lower()
        value = re.sub(r"[–—‐‒−]", "-", value)
        value = re.sub(r"\s+", " ", value)
        value = value.replace("м", "m")
        return value
    return value


def load_dataset(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Input dataset not found: {filepath}")
    return pd.read_csv(filepath)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = result.columns.str.strip().str.lower()
    result = result.map(normalize_text)
    return result


def harmonize_educational_program(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .map(normalize_text)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def harmonize_column(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.astype(str).map(normalize_text).replace(mapping)


def harmonize_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies harmonization rules and returns:
    1) harmonized dataframe
    2) harmonization report dataframe
    """
    result = df.copy()
    report_rows = []

    for col, mapping in HARMONIZATION_RULES.items():
        if col not in result.columns:
            continue

        before = result[col].copy()
        result[col] = harmonize_column(result[col], mapping)
        changed_mask = before.astype(str).map(normalize_text) != result[col].astype(
            str
        ).map(normalize_text)
        changed_count = int(changed_mask.sum())

        report_rows.append(
            {
                "column": col,
                "changed_rows": changed_count,
                "mapping_rules": str(mapping),
            }
        )

    if "educational_program" in result.columns:
        before = result["educational_program"].copy()
        result["educational_program"] = harmonize_educational_program(
            result["educational_program"]
        )
        changed_mask = (
            before.astype(str).map(normalize_text) != result["educational_program"]
        )
        changed_count = int(changed_mask.sum())

        report_rows.append(
            {
                "column": "educational_program",
                "changed_rows": changed_count,
                "mapping_rules": "normalized case, spaces, dashes, and cyrillic/latin code variants",
            }
        )

    report_df = pd.DataFrame(report_rows)
    return result, report_df


def validate_canonical_values(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns unexpected values that remain after harmonization.
    """
    unexpected: dict[str, list[str]] = {}

    for col, allowed_values in CANONICAL_VALUES.items():
        if col not in df.columns:
            continue

        observed = set(
            df[col].dropna().astype(str).map(normalize_text).unique().tolist()
        )
        invalid = sorted(v for v in observed if v not in allowed_values)

        if invalid:
            unexpected[col] = invalid

    if "educational_program" in df.columns:
        observed_programs = set(
            df["educational_program"]
            .dropna()
            .astype(str)
            .map(normalize_text)
            .unique()
            .tolist()
        )
        invalid_programs = sorted(
            v for v in observed_programs if v not in CANONICAL_EDUCATIONAL_PROGRAMS
        )

        if invalid_programs:
            unexpected["educational_program"] = invalid_programs

    return unexpected


def save_unexpected_values_report(
    unexpected: dict[str, list[str]], output_file: Path
) -> None:
    lines: list[str] = []

    lines.append("=== UNEXPECTED VALUES AFTER HARMONIZATION ===")
    if not unexpected:
        lines.append("No unexpected values detected.")
    else:
        for col, values in unexpected.items():
            lines.append("")
            lines.append(f"COLUMN: {col}")
            for value in values:
                lines.append(f"  - {value}")

    output_file.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Target construction
# ============================================================


def build_surrogate_academic_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds binary surrogate/proxy label for academic risk.

    Rule:
    academic_risk = 1 if GPA is below 2.7 OR attendance is below 70%
    academic_risk = 0 otherwise
    """
    result = df.copy()

    required_cols = ["average_gpa", "average_attendance"]
    missing_required = [col for col in required_cols if col not in result.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns for target construction: {missing_required}"
        )

    def risk_rule(row: pd.Series) -> int | None:
        gpa = row["average_gpa"]
        attendance = row["average_attendance"]

        if pd.isna(gpa) or pd.isna(attendance):
            return None

        if gpa == "below 2.7":
            return 1

        if attendance == "below 70%":
            return 1

        return 0

    result[TARGET_COLUMN] = result.apply(risk_rule, axis=1)

    return result


def drop_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result[result[TARGET_COLUMN].notna()].copy()
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(int)
    return result


# ============================================================
# Scenario construction
# ============================================================


def build_feature_manifest() -> pd.DataFrame:
    rows = []

    for feature in SURVEY_FEATURE_COLUMNS:
        if feature in NOMINAL_FEATURES:
            feature_type = "nominal_categorical"
        elif feature in ORDINAL_FEATURES:
            feature_type = "ordinal_categorical"
        else:
            feature_type = "categorical"

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "used_in_scenario_a": feature in SCENARIO_A_FEATURES,
                "used_in_scenario_b": feature in SCENARIO_B_FEATURES,
                "is_target": False,
            }
        )

    rows.append(
        {
            "feature": TARGET_COLUMN,
            "feature_type": "binary_target",
            "used_in_scenario_a": True,
            "used_in_scenario_b": True,
            "is_target": True,
        }
    )

    return pd.DataFrame(rows)


def build_scenario_dataset(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    missing_cols = [
        col for col in feature_columns + [TARGET_COLUMN] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing columns for scenario dataset: {missing_cols}")

    return df[feature_columns + [TARGET_COLUMN]].copy()


# ============================================================
# Summary and reporting
# ============================================================


def class_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    total = len(df)

    rows = []
    for label, count in counts.items():
        rows.append(
            {
                "academic_risk": label,
                "count": int(count),
                "percent": round(count / total * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def save_preparation_summary(
    full_df: pd.DataFrame,
    scenario_a_df: pd.DataFrame,
    scenario_b_df: pd.DataFrame,
    unexpected: dict[str, list[str]],
    output_file: Path,
) -> None:
    lines: list[str] = []

    lines.append("=== MODEL PREPARATION SUMMARY ===")
    lines.append("")
    lines.append(f"Full labeled dataset shape: {full_df.shape}")
    lines.append(f"Scenario A dataset shape: {scenario_a_df.shape}")
    lines.append(f"Scenario B dataset shape: {scenario_b_df.shape}")
    lines.append("")
    lines.append(
        f"Scenario A feature count (without target): {len(SCENARIO_A_FEATURES)}"
    )
    lines.append(
        f"Scenario B feature count (without target): {len(SCENARIO_B_FEATURES)}"
    )
    lines.append("")
    lines.append("Class balance:")
    lines.append(class_balance_table(full_df).to_string(index=False))
    lines.append("")

    if unexpected:
        lines.append("Unexpected values remain after harmonization:")
        for col, values in unexpected.items():
            lines.append(f"- {col}: {values}")
    else:
        lines.append("No unexpected values remain after harmonization.")

    output_file.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main pipeline
# ============================================================


def run_model_preparation_pipeline() -> None:
    raw_df = load_dataset(INPUT_FILE)
    df = standardize_dataframe(raw_df)

    harmonized_df, harmonization_report = harmonize_categories(df)
    unexpected_values = validate_canonical_values(harmonized_df)

    labeled_df = build_surrogate_academic_risk(harmonized_df)
    labeled_df = drop_rows_with_missing_target(labeled_df)

    full_output_df = labeled_df.copy()

    modeling_df = labeled_df.drop(
        columns=[col for col in TECHNICAL_COLUMNS if col in labeled_df.columns],
        errors="ignore",
    )

    scenario_a_df = build_scenario_dataset(modeling_df, SCENARIO_A_FEATURES)
    scenario_b_df = build_scenario_dataset(modeling_df, SCENARIO_B_FEATURES)

    feature_manifest_df = build_feature_manifest()

    full_output_df.to_csv(FULL_OUTPUT_FILE, index=False)
    scenario_a_df.to_csv(SCENARIO_A_FILE, index=False)
    scenario_b_df.to_csv(SCENARIO_B_FILE, index=False)
    feature_manifest_df.to_csv(FEATURE_MANIFEST_FILE, index=False)
    harmonization_report.to_csv(HARMONIZATION_REPORT_FILE, index=False)

    save_unexpected_values_report(unexpected_values, UNEXPECTED_VALUES_REPORT_FILE)
    save_preparation_summary(
        full_df=full_output_df,
        scenario_a_df=scenario_a_df,
        scenario_b_df=scenario_b_df,
        unexpected=unexpected_values,
        output_file=PREPARATION_SUMMARY_FILE,
    )

    print("\n===== MODEL PREPARATION COMPLETED =====")
    print(f"Input file: {INPUT_FILE}")
    print(f"Full harmonized labeled dataset: {FULL_OUTPUT_FILE}")
    print(f"Scenario A dataset: {SCENARIO_A_FILE}")
    print(f"Scenario B dataset: {SCENARIO_B_FILE}")
    print(f"Feature manifest: {FEATURE_MANIFEST_FILE}")
    print(f"Harmonization report: {HARMONIZATION_REPORT_FILE}")
    print(f"Unexpected values report: {UNEXPECTED_VALUES_REPORT_FILE}")
    print(f"Preparation summary: {PREPARATION_SUMMARY_FILE}")

    print("\nClass balance:")
    print(class_balance_table(full_output_df).to_string(index=False))

    if unexpected_values:
        print("\nUnexpected values remain after harmonization:")
        for col, values in unexpected_values.items():
            print(f"- {col}: {values}")
    else:
        print("\nNo unexpected values remain after harmonization.")


def main() -> None:
    run_model_preparation_pipeline()


if __name__ == "__main__":
    main()
