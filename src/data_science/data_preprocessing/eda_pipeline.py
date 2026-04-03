from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import matplotlib

# Try to use an interactive backend for VSCode GUI.
# If TkAgg is unavailable, matplotlib will keep the default backend.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# Paths and global configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (BASE_DIR / "../data/output/processed_students_unlabeled.csv").resolve()

EDA_DIR = (BASE_DIR / "../data/output/eda").resolve()
FIGURES_DIR = EDA_DIR / "figures"
TABLES_DIR = EDA_DIR / "tables"

EDA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SHOW_PLOTS = False
SAVE_FIGURES = True
INCLUDE_MISSING_IN_PLOTS = False

TECHNICAL_COLUMNS = ["student_id", "role", "username", "password"]


# ============================================================
# Feature groups
# ============================================================

RANGE_COLUMNS = [
    "age",
    "average_gpa",
    "average_attendance",
    "self_study_hours",
    "average_sleep_hours",
    "exam_preparation_hours",
]

ORDINAL_MAPPINGS: dict[str, dict[str, float]] = {
    "motivation_level": {
        "very low": 1,
        "low": 2,
        "average": 3,
        "high": 4,
        "very high": 5,
    },
    "stress_frequency": {
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
        "very often": 5,
    },
    "importance_of_interest_in_subject": {
        "not important": 1,
        "slightly important": 2,
        "moderately important": 3,
        "important": 4,
        "very important": 5,
    },
    "teacher_competence_satisfaction": {
        "not satisfied at all": 1,
        "rather dissatisfied": 2,
        "hard to say": 3,
        "rather satisfied": 4,
        "fully satisfied": 5,
    },
    "use_of_university_resources": {
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
        "very often": 5,
    },
    "financial_support_from_family": {
        "no financial support": 0,
        "partial support": 1,
        "full support": 2,
    },
    "has_job": {
        "no": 0,
        "yes, part-time": 1,
        "yes, full-time": 2,
    },
    "communication_skills": {
        "very low": 1,
        "low": 2,
        "average": 3,
        "high": 4,
        "very high": 5,
    },
    "gender": {
        "male": 0,
        "female": 1,
    },
    "degree": {
        "bachelor": 1,
        "master": 2,
        "phd": 3,
    },
    "year_of_study": {
        "1 year": 1,
        "2 year": 2,
        "3 year": 3,
    },
}


# ============================================================
# Plot configuration
# ============================================================


def configure_plot_style() -> None:
    plt.style.use("default")
    sns.set_theme(style="whitegrid", context="talk")


def finalize_figure(fig: plt.Figure, filename: str) -> None:
    if SAVE_FIGURES:
        fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show(block=True)
    else:
        plt.close(fig)


def save_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(TABLES_DIR / filename, index=False)


# ============================================================
# Text normalization and loading
# ============================================================


def normalize_text_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip().lower()
        value = re.sub(r"[–—‐‒−]", "-", value)
        value = value.replace("м", "m")  # normalize occasional Cyrillic 'м'
        return value
    return value


def load_processed_dataset(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found: {filepath}")
    return pd.read_csv(filepath)


def drop_technical_columns(
    df: pd.DataFrame, technical_columns: list[str]
) -> pd.DataFrame:
    return df.drop(
        columns=[col for col in technical_columns if col in df.columns], errors="ignore"
    )


def standardize_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = result.columns.str.strip().str.lower()
    result = result.map(normalize_text_value)
    return result


# ============================================================
# Numeric feature construction
# ============================================================


def parse_range_text(value: Any) -> float:
    """
    Converts interval-like text values into approximate numeric values.

    Examples:
    - '3.1-3.2' -> 3.15
    - '80-89%' -> 84.5
    - '1-3 hours' -> 2.0
    - 'less than 1 hour' -> 0.5
    - 'above 3.5' -> 3.6
    - '11-20 hours' -> 15.5
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()
    s = re.sub(r"[–—‐‒−]", "-", s)

    if s in {"do not study", "do not prepare"}:
        return 0.0

    s = s.replace("%", "").replace("hours", "").replace("hour", "").strip()

    if "less than" in s:
        num = re.findall(r"[0-9.]+", s)
        if num:
            return float(num[0]) * 0.5

    if "more than" in s:
        num = re.findall(r"[0-9.]+", s)
        if num:
            return float(num[0]) + 1.0

    if "above" in s:
        num = re.findall(r"[0-9.]+", s)
        if num:
            base = float(num[0])
            if base >= 10:
                return base + 1.0
            return base + 0.1

    if "-" in s:
        parts = s.split("-")
        try:
            a = float(parts[0].strip())
            b = float(parts[1].strip())
            return (a + b) / 2.0
        except ValueError:
            return np.nan

    num = re.findall(r"[0-9.]+", s)
    if num:
        return float(num[0])

    return np.nan


def safe_series_map(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(mapping).astype("float")


def add_numeric_range_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in RANGE_COLUMNS:
        if col in result.columns:
            result[f"{col}_num"] = result[col].apply(parse_range_text)
    return result


def add_ordinal_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in result.columns:
            result[f"{col}_num"] = safe_series_map(result[col], mapping)
    return result


def prepare_eda_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = drop_technical_columns(df, TECHNICAL_COLUMNS)
    result = standardize_text_frame(result)
    result = add_numeric_range_features(result)
    result = add_ordinal_numeric_features(result)
    return result


# ============================================================
# Basic overview and summary tables
# ============================================================


def print_basic_overview(df: pd.DataFrame) -> None:
    print("\n===== DATASET OVERVIEW =====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes.sort_index())


def build_dataset_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "number_of_rows",
                "number_of_columns",
                "number_of_numeric_columns",
                "number_of_categorical_columns",
            ],
            "value": [
                df.shape[0],
                df.shape[1],
                df.select_dtypes(include=[np.number]).shape[1],
                df.select_dtypes(exclude=[np.number]).shape[1],
            ],
        }
    )


def build_feature_cardinality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        non_null = int(df[col].notna().sum())
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))
        dtype = str(df[col].dtype)

        mode_series = df[col].mode(dropna=True)
        most_common = mode_series.iloc[0] if not mode_series.empty else np.nan

        rows.append(
            {
                "feature": col,
                "dtype": dtype,
                "non_null_count": non_null,
                "missing_count": missing,
                "unique_values": unique,
                "most_common_value": most_common,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(by=["dtype", "feature"]).reset_index(drop=True)
    )


def build_missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    result = pd.DataFrame(
        {
            "feature": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": missing_pct.values,
        }
    )

    result = result[result["missing_count"] > 0]
    result = result.sort_values(
        by=["missing_count", "feature"], ascending=[False, True]
    )
    return result.reset_index(drop=True)


def build_program_level_summary_table(
    df: pd.DataFrame, top_n: int = 15
) -> pd.DataFrame:
    required_columns = [
        "educational_program",
        "average_gpa_num",
        "average_attendance_num",
        "self_study_hours_num",
        "average_sleep_hours_num",
    ]
    if not all(col in df.columns for col in required_columns):
        return pd.DataFrame()

    summary = (
        df.groupby("educational_program")
        .agg(
            student_count=("educational_program", "size"),
            median_gpa=("average_gpa_num", "median"),
            median_attendance=("average_attendance_num", "median"),
            median_self_study=("self_study_hours_num", "median"),
            median_sleep=("average_sleep_hours_num", "median"),
        )
        .sort_values(by=["student_count", "median_gpa"], ascending=[False, False])
        .head(top_n)
        .reset_index()
    )
    return summary


# ============================================================
# Plot helpers
# ============================================================


def get_plotting_series(
    df: pd.DataFrame, col: str, include_missing: bool = False
) -> pd.Series:
    series = df[col].copy()
    if include_missing:
        return series.fillna("missing")
    return series.dropna()


def get_descending_count_order(
    df: pd.DataFrame, col: str, include_missing: bool = False
) -> list[str]:
    series = get_plotting_series(df, col, include_missing=include_missing)
    return series.value_counts().sort_values(ascending=False).index.tolist()


def annotate_bar_counts(ax, orientation: str = "vertical") -> None:
    for patch in ax.patches:
        if orientation == "vertical":
            height = patch.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(round(height))}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 4),
                    textcoords="offset points",
                )
        else:
            width = patch.get_width()
            if width > 0:
                ax.annotate(
                    f"{int(round(width))}",
                    (width, patch.get_y() + patch.get_height() / 2),
                    ha="left",
                    va="center",
                    fontsize=10,
                    xytext=(4, 0),
                    textcoords="offset points",
                )


def annotate_bar_values(ax, orientation: str = "vertical", decimals: int = 2) -> None:
    for patch in ax.patches:
        if orientation == "vertical":
            height = patch.get_height()
            if pd.notna(height):
                ax.annotate(
                    f"{height:.{decimals}f}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 4),
                    textcoords="offset points",
                )
        else:
            width = patch.get_width()
            if pd.notna(width):
                ax.annotate(
                    f"{width:.{decimals}f}",
                    (width, patch.get_y() + patch.get_height() / 2),
                    ha="left",
                    va="center",
                    fontsize=10,
                    xytext=(4, 0),
                    textcoords="offset points",
                )


# ============================================================
# EDA plots
# ============================================================


def plot_missing_values(df: pd.DataFrame) -> None:
    missing_table = build_missing_values_table(df)

    if missing_table.empty:
        print("\nNo missing values detected.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=missing_table.sort_values("missing_count", ascending=False),
        x="missing_count",
        y="feature",
        ax=ax,
    )
    annotate_bar_counts(ax, orientation="horizontal")
    ax.set_title("Missing Values per Feature")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    finalize_figure(fig, "missing_values_barplot.png")


def plot_demographic_profile(df: pd.DataFrame) -> None:
    demographic_columns = ["gender", "age", "degree", "year_of_study"]
    available_cols = [col for col in demographic_columns if col in df.columns]

    if not available_cols:
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for ax, col in zip(axes, available_cols):
        plot_df = df.copy()
        if not INCLUDE_MISSING_IN_PLOTS:
            plot_df = plot_df[plot_df[col].notna()]

        order = get_descending_count_order(
            plot_df, col, include_missing=INCLUDE_MISSING_IN_PLOTS
        )

        sns.countplot(data=plot_df, x=col, order=order, ax=ax)
        annotate_bar_counts(ax, orientation="vertical")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)

    for idx in range(len(available_cols), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Demographic Profile of Respondents", y=1.02)
    fig.tight_layout()
    finalize_figure(fig, "demographic_profile_dashboard.png")


def plot_program_and_year_profile(df: pd.DataFrame, top_n_programs: int = 12) -> None:
    if "educational_program" not in df.columns or "year_of_study" not in df.columns:
        return

    df_temp = df.copy()
    if not INCLUDE_MISSING_IN_PLOTS:
        df_temp = df_temp[
            df_temp["educational_program"].notna() & df_temp["year_of_study"].notna()
        ]

    top_programs = (
        df_temp["educational_program"]
        .value_counts()
        .head(top_n_programs)
        .index.tolist()
    )
    df_top = df_temp[df_temp["educational_program"].isin(top_programs)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    row_order = (
        df_top["educational_program"]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.countplot(
        data=df_top,
        y="educational_program",
        order=row_order,
        ax=axes[0],
    )
    annotate_bar_counts(axes[0], orientation="horizontal")
    axes[0].set_title(
        f"Top {top_n_programs} Educational Programs by Number of Respondents"
    )
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")

    crosstab = pd.crosstab(df_top["educational_program"], df_top["year_of_study"])
    crosstab = crosstab.reindex(index=row_order)

    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Student Count"},
        ax=axes[1],
    )
    axes[1].set_title("Educational Program × Year of Study")
    axes[1].set_xlabel("Year of Study")
    axes[1].set_ylabel("")

    fig.tight_layout()
    finalize_figure(fig, "program_year_profile.png")


def plot_academic_distributions(df: pd.DataFrame) -> None:
    plot_columns = [
        "average_gpa",
        "average_attendance",
        "self_study_hours",
        "average_sleep_hours",
        "exam_preparation_hours",
    ]
    available_cols = [col for col in plot_columns if col in df.columns]

    if not available_cols:
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for ax, col in zip(axes, available_cols):
        plot_df = df.copy()
        if not INCLUDE_MISSING_IN_PLOTS:
            plot_df = plot_df[plot_df[col].notna()]

        order = get_descending_count_order(
            plot_df, col, include_missing=INCLUDE_MISSING_IN_PLOTS
        )

        sns.countplot(data=plot_df, x=col, order=order, ax=ax)
        annotate_bar_counts(ax, orientation="vertical")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    for idx in range(len(available_cols), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Distribution of Key Academic Indicators", y=1.02)
    fig.tight_layout()
    finalize_figure(fig, "academic_distributions_dashboard.png")


def plot_behavioral_distributions(df: pd.DataFrame) -> None:
    plot_columns = [
        "motivation_level",
        "stress_frequency",
        "financial_support_from_family",
        "has_job",
        "communication_skills",
        "use_of_university_resources",
    ]
    available_cols = [col for col in plot_columns if col in df.columns]

    if not available_cols:
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for ax, col in zip(axes, available_cols):
        plot_df = df.copy()
        if not INCLUDE_MISSING_IN_PLOTS:
            plot_df = plot_df[plot_df[col].notna()]

        order = get_descending_count_order(
            plot_df, col, include_missing=INCLUDE_MISSING_IN_PLOTS
        )

        sns.countplot(data=plot_df, x=col, order=order, ax=ax)
        annotate_bar_counts(ax, orientation="vertical")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    for idx in range(len(available_cols), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Distribution of Behavioral and Contextual Factors", y=1.02)
    fig.tight_layout()
    finalize_figure(fig, "behavioral_distributions_dashboard.png")


def summarize_outcome_by_factor(
    df: pd.DataFrame,
    factor_col: str,
    outcome_col: str,
) -> pd.DataFrame:
    if factor_col not in df.columns or outcome_col not in df.columns:
        return pd.DataFrame()

    temp_df = df[[factor_col, outcome_col]].copy()

    if not INCLUDE_MISSING_IN_PLOTS:
        temp_df = temp_df[temp_df[factor_col].notna() & temp_df[outcome_col].notna()]
    else:
        temp_df[factor_col] = temp_df[factor_col].fillna("missing")

    summary = (
        temp_df.groupby(factor_col, dropna=False)[outcome_col]
        .agg(["median", "mean", "count"])
        .reset_index()
        .rename(columns={"median": "median_value", "mean": "mean_value", "count": "n"})
    )

    summary = summary.sort_values(
        by=["median_value", "n"], ascending=[False, False]
    ).reset_index(drop=True)
    return summary


def plot_outcome_by_key_factors(
    df: pd.DataFrame,
    outcome_col: str,
    outcome_label: str,
    filename: str,
) -> None:
    factor_cols = [
        "motivation_level",
        "stress_frequency",
        "has_job",
        "financial_support_from_family",
        "use_of_university_resources",
        "exam_preparation_hours",
    ]
    available_factors = [
        col for col in factor_cols if col in df.columns and outcome_col in df.columns
    ]

    if not available_factors:
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for ax, factor in zip(axes, available_factors):
        summary = summarize_outcome_by_factor(df, factor, outcome_col)
        if summary.empty:
            continue

        sns.barplot(data=summary, x=factor, y="median_value", ax=ax)
        annotate_bar_values(ax, orientation="vertical", decimals=2)
        ax.set_title(f"{outcome_label} by {factor.replace('_', ' ').title()}")
        ax.set_xlabel("")
        ax.set_ylabel(f"Median {outcome_label}")
        ax.tick_params(axis="x", rotation=30)

    for idx in range(len(available_factors), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    finalize_figure(fig, filename)


# ============================================================
# Correlation analysis
# ============================================================


def build_correlation_feature_list(df: pd.DataFrame) -> list[str]:
    candidate_features = [
        "age_num",
        "average_gpa_num",
        "average_attendance_num",
        "self_study_hours_num",
        "average_sleep_hours_num",
        "exam_preparation_hours_num",
        "motivation_level_num",
        "stress_frequency_num",
        "importance_of_interest_in_subject_num",
        "teacher_competence_satisfaction_num",
        "use_of_university_resources_num",
        "has_job_num",
        "financial_support_from_family_num",
        "communication_skills_num",
        "gender_num",
        "degree_num",
        "year_of_study_num",
    ]
    return [col for col in candidate_features if col in df.columns]


def build_spearman_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr_features = build_correlation_feature_list(df)
    if not corr_features:
        return pd.DataFrame()
    return df[corr_features].corr(method="spearman")


def build_top_correlations_table(
    corr_matrix: pd.DataFrame,
    target_col: str,
    top_n: int = 8,
) -> pd.DataFrame:
    if corr_matrix.empty or target_col not in corr_matrix.columns:
        return pd.DataFrame()

    series = corr_matrix[target_col].drop(labels=[target_col]).dropna()
    result = (
        series.abs()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("abs_correlation")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    result["signed_correlation"] = result["feature"].map(series.to_dict())
    return result[["feature", "signed_correlation", "abs_correlation"]]


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    if corr_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Spearman Correlation Heatmap")
    finalize_figure(fig, "spearman_correlation_heatmap.png")


def plot_focused_target_heatmap(corr_matrix: pd.DataFrame) -> None:
    """
    Compact heatmap showing how numeric EDA features correlate
    specifically with GPA and Attendance.
    """
    if corr_matrix.empty:
        return

    target_cols = [
        col
        for col in ["average_gpa_num", "average_attendance_num"]
        if col in corr_matrix.columns
    ]
    if not target_cols:
        return

    focused = corr_matrix[target_cols].drop(index=target_cols, errors="ignore")
    if focused.empty:
        return

    focused = focused.reindex(
        focused.abs().max(axis=1).sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(focused) * 0.45)))
    sns.heatmap(
        focused,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlations of Features with GPA and Attendance")
    finalize_figure(fig, "focused_gpa_attendance_heatmap.png")


def plot_top_correlations_with_target(
    corr_matrix: pd.DataFrame,
    target_col: str,
    target_label: str,
    filename: str,
    top_n: int = 8,
) -> None:
    top_corr = build_top_correlations_table(corr_matrix, target_col, top_n=top_n)
    if top_corr.empty:
        return

    plot_df = top_corr.sort_values(by="abs_correlation", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=plot_df, x="signed_correlation", y="feature", ax=ax)
    annotate_bar_values(ax, orientation="horizontal", decimals=2)
    ax.set_title(f"Top Spearman Correlations with {target_label}")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("")
    finalize_figure(fig, filename)


# ============================================================
# Main EDA pipeline
# ============================================================


def run_eda_pipeline() -> None:
    configure_plot_style()

    raw_df = load_processed_dataset(INPUT_FILE)
    df = prepare_eda_dataset(raw_df)

    print_basic_overview(df)

    # ----------------------------
    # Tables
    # ----------------------------
    overview_table = build_dataset_overview_table(df)
    feature_cardinality_table = build_feature_cardinality_table(df)
    missing_values_table = build_missing_values_table(df)
    program_summary_table = build_program_level_summary_table(df, top_n=15)

    save_table(overview_table, "dataset_overview.csv")
    save_table(feature_cardinality_table, "feature_cardinality_summary.csv")

    if not missing_values_table.empty:
        save_table(missing_values_table, "missing_values_summary.csv")

    if not program_summary_table.empty:
        save_table(program_summary_table, "program_level_academic_summary.csv")

    # ----------------------------
    # Correlations
    # ----------------------------
    corr_matrix = build_spearman_correlation_matrix(df)

    if not corr_matrix.empty:
        corr_table = (
            corr_matrix.round(3).reset_index().rename(columns={"index": "feature"})
        )
        save_table(corr_table, "spearman_correlation_matrix.csv")

        top_gpa_corr = build_top_correlations_table(
            corr_matrix, "average_gpa_num", top_n=8
        )
        top_att_corr = build_top_correlations_table(
            corr_matrix, "average_attendance_num", top_n=8
        )

        if not top_gpa_corr.empty:
            save_table(top_gpa_corr, "top_correlations_with_gpa.csv")
            print("\nTop correlations with average_gpa_num:")
            print(top_gpa_corr)

        if not top_att_corr.empty:
            save_table(top_att_corr, "top_correlations_with_attendance.csv")
            print("\nTop correlations with average_attendance_num:")
            print(top_att_corr)

    # ----------------------------
    # Plots
    # ----------------------------
    plot_missing_values(df)
    plot_demographic_profile(df)
    plot_program_and_year_profile(df, top_n_programs=12)
    plot_academic_distributions(df)
    plot_behavioral_distributions(df)

    plot_outcome_by_key_factors(
        df=df,
        outcome_col="average_gpa_num",
        outcome_label="GPA",
        filename="gpa_by_key_factors.png",
    )

    plot_outcome_by_key_factors(
        df=df,
        outcome_col="average_attendance_num",
        outcome_label="Attendance",
        filename="attendance_by_key_factors.png",
    )

    plot_correlation_heatmap(corr_matrix)
    plot_focused_target_heatmap(corr_matrix)

    plot_top_correlations_with_target(
        corr_matrix=corr_matrix,
        target_col="average_gpa_num",
        target_label="GPA",
        filename="top_correlations_with_gpa.png",
        top_n=8,
    )

    plot_top_correlations_with_target(
        corr_matrix=corr_matrix,
        target_col="average_attendance_num",
        target_label="Attendance",
        filename="top_correlations_with_attendance.png",
        top_n=8,
    )

    print("\nEDA pipeline completed.")
    print(f"Input dataset: {INPUT_FILE}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Tables directory: {TABLES_DIR}")


def main() -> None:
    run_eda_pipeline()


if __name__ == "__main__":
    main()
