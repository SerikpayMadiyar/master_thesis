import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from drop_columns import DROP_COLUMNS_PROCESSED

def starting_point(df):
    print("BASIC INFO AFTER DROPPING TECHNICAL COLUMNS")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

def parse_range_text(value):
    """
    Converts strings like:
    - '3.1-3.2' -> midpoint
    - '80-89%' -> midpoint / 100 if %
    - '1-3 hours' -> midpoint
    - 'less than 1 hour' -> 0.5
    - 'more than 4 hours' -> 5 (approx)
    - 'above 3.5' -> 3.6 (approx)
    - '20-21' (age) -> midpoint 20.5
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    s = s.replace("–", "-")  

    has_percent = "%" in s
    s = s.replace("%", "").replace("hours", "").replace("hour", "").strip()

    if "less than" in s:
        num = re.findall(r"[0-9.]+", s)
        if num:
            val = float(num[0]) * 0.5
            return val / 100 if has_percent else val
    if "more than" in s:
        num = re.findall(r"[0-9.]+", s)
        if num:
            val = float(num[0]) + 1.0
            return val / 100 if has_percent else val
    if "above" in s:
        num = re.findall(r"[0-9.]+", s)
        if num:
            val = float(num[0]) + 0.1
            return val / 100 if has_percent else val

    if "-" in s:
        parts = s.split("-")
        try:
            a = float(parts[0])
            b = float(parts[1])
            val = (a + b) / 2.0
            return val / 100 if has_percent else val
        except Exception:
            pass

    num = re.findall(r"[0-9.]+", s)
    if num:
        val = float(num[0])
        return val / 100 if has_percent else val

    return np.nan

def map_safe(series, mapping):
    return series.str.lower().map(mapping).astype("float")

def mapping(df):
    motivation_map = {"low": 1, "average": 2, "high": 3}
    stress_map = {"rarely": 1, "sometimes": 2, "often": 3}
    importance_map = {"not important": 1, "moderately important": 2, "important": 3}
    satisfaction_map = {"not satisfied": 1, "hard to say": 2, "rather satisfied": 3}
    use_resources_map = {"rarely": 1, "sometimes": 2, "often": 3}
    job_map = {"no": 0, "yes, part-time": 1, "yes, full-time": 2}
    support_map = {"no support": 0, "partial support": 1, "full support": 2}
    comm_map = {"low": 1, "average": 2, "high": 3}

    if "motivation_level" in df.columns:
        df["motivation_num"] = map_safe(df["motivation_level"], motivation_map)

    if "stress_frequency" in df.columns:
        df["stress_num"] = map_safe(df["stress_frequency"], stress_map)

    if "importance_of_interest_in_subject" in df.columns:
        df["interest_importance_num"] = map_safe(df["importance_of_interest_in_subject"], importance_map)

    if "teacher_competence_satisfaction" in df.columns:
        df["teacher_satisfaction_num"] = map_safe(df["teacher_competence_satisfaction"], satisfaction_map)

    if "use_of_university_resources" in df.columns:
        df["resources_use_num"] = map_safe(df["use_of_university_resources"], use_resources_map)

    if "has_job" in df.columns:
        df["job_num"] = df["has_job"].str.lower().map(job_map).astype("float")

    if "financial_support_from_family" in df.columns:
        df["family_support_num"] = df["financial_support_from_family"].str.lower().map(support_map).astype("float")

    if "communication_skills" in df.columns:
        df["comm_skills_num"] = map_safe(df["communication_skills"], comm_map)

    if "gender" in df.columns:
        df["gender_num"] = df["gender"].str.lower().map({"male": 0, "female": 1}).astype("float")
        
    return df

def student_and_gender(df):
    print("\nEDA PART 1: STUDENT COUNT AND GENDER")

    total_students = len(df)
    print(f"Total number of students: {total_students}")

    if "gender" in df.columns:
        gender_counts = df["gender"].value_counts()
        gender_percent = gender_counts / total_students * 100

        print("\nGender distribution (counts):")
        print(gender_counts)
        print("\nGender distribution (percent):")
        print(gender_percent.round(2))

        dominant_gender = gender_counts.idxmax()
        dom_pct = gender_percent.loc[dominant_gender]

        plt.figure(figsize=(20, 10))
        plt.pie(
            gender_counts,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"linewidth": 1, "edgecolor": "white"}
        )
        plt.title("Gender Distribution of Respondents")
        plt.tight_layout()
        plt.show()

    else:
        print("Column 'gender' not found.")

def age_of_students(df):
    print("\nEDA PART 2: AGE OF STUDENTS")

    if "age_num" in df.columns:
        print("Summary statistics for age:")
        print(df["age_num"].describe())

        mean_age = df["age_num"].mean()
        min_age = df["age_num"].min()
        max_age = df["age_num"].max()

        plt.figure(figsize=(20, 10))
        df["age"].value_counts().sort_index().plot(kind="bar")
        plt.title("Age Range Distribution")
        plt.xlabel("Age Range")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'age' / 'age_num' not found for age analysis.")

def program_and_course(df):
    print("\nEDA PART 3: PROGRAM AND COURSE")

    if "educational_program" in df.columns and "year_of_study" in df.columns:
        program_course_table = pd.crosstab(df["educational_program"],
                                        df["year_of_study"])

        print("\nNumber of students per educational program and year of study:")
        print(program_course_table)

        plt.figure(figsize=(20, 10))
        sns.heatmap(program_course_table,
                    annot=True, fmt="d", cmap="Blues",
                    cbar_kws={"label": "Number of students"})
        plt.title("Students per Educational Program and Year of Study")
        plt.xlabel("Year of Study")
        plt.ylabel("Educational Program")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 10))
        sns.countplot(data=df,
                    x="educational_program",
                    hue="year_of_study")
        plt.title("Students per Educational Program and Year of Study")
        plt.xlabel("Educational Program")
        plt.ylabel("Number of Students")
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Year of Study")
        plt.tight_layout()
        plt.show()

    else:
        print("Columns 'educational_program' and/or 'year_of_study' not found.")

def correlation_heatmap(df, corr_matrix):
    print("\nPART 4: CORRELATION HEATMAP")
  
    print("Spearman Correlation Matrix:")
    print(corr_matrix.round(2))

    plt.figure(figsize=(20, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    for target in ["average_gpa_num", "motivation_num", "stress_num"]:
        if target in corr_matrix.columns:
            print(f"\nTop correlations with {target}:")
            print(
                corr_matrix[target]
                .drop(target)
                .sort_values(ascending=False)
                .head(5)
                .round(2)
            )

def sorted_correlation_heatmap(df, corr_matrix):
    print("\nPART 5: SORTED AND MASKED UPPER TRIANGLE CORRELATION HEATMAP")

    if "average_gpa_num" in corr_matrix.columns:
        order = corr_matrix["average_gpa_num"].abs().sort_values(ascending=False).index
        corr_sorted = corr_matrix.loc[order, order]
    else:
        corr_sorted = corr_matrix

    mask = np.triu(np.ones_like(corr_sorted, dtype=bool))

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        corr_sorted,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        square=True
    )
    plt.title("Sorted and Masked Upper Triangle Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def median_of_numerical_features(df):
    print("\n===== PART A: MEDIAN OF KEY ACADEMIC & BEHAVIORAL INDICATORS =====")

    # Normalize dashes globally to avoid Unicode mismatch problems
    df = df.replace({r'[–—‐-‒−]': '-'}, regex=True)

    # === Mapping dictionaries ===
    age_map = {
        "16-17": 16.5, "18-19": 18.5, "20-21": 20.5,
        "22-23": 22.5, "above 24": 25
    }

    gpa_map = {
        "below 2.7": 2.6, "2.8-3.0": 2.9, "3.1-3.2": 3.15,
        "3.3-3.4": 3.35, "above 3.5": 3.6
    }

    attendance_map = {
        "below 70%": 65, "70-79%": 74.5,
        "80-89%": 84.5, "90-100%": 95
    }

    study_map = {
        "do not study": 0, "less than 1 hour": 1,
        "1-3 hours": 2, "more than 4 hours": 3
    }

    sleep_map = {
        "less than 6 hours": 5, "7-9 hours": 8, "more than 10 hours": 11
    }

    prep_map = {
        "do not prepare": 0, "less than 1 hour": 1,
        "2-8 hours": 5, "9-15 hours": 12,
        "more than 16 hours": 16
    }

    # === Conversion ===
    df["age_num"] = df["age"].map(age_map)
    df["gpa_num"] = df["average_gpa"].map(gpa_map)
    df["attendance_num"] = df["average_attendance"].map(attendance_map)
    df["study_num"] = df["self_study_hours"].map(study_map)
    df["sleep_num"] = df["average_sleep_hours"].map(sleep_map)
    df["prep_num"] = df["exam_preparation_hours"].map(prep_map)

    # === Summary Table ===
    median_df = pd.DataFrame({
        "Feature": ["Age", "GPA", "Attendance", "Self Study", "Sleep", "Exam Prep"],
        "Median": [
            df["age_num"].median(),
            df["gpa_num"].median(),
            df["attendance_num"].median(),
            df["study_num"].median(),
            df["sleep_num"].median(),
            df["prep_num"].median()
        ]
    })

    print("\n---- Median of Converted Numerical Features ----")
    print(median_df.to_string(index=False))

    # === Visualization ===
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(
        data=median_df,
        x="Median",
        y="Feature",
        palette="viridis"
    )

    # Add median labels on the right of each bar
    for i, val in enumerate(median_df["Median"]):
        ax.text(
            val + 0.1,      # Slightly to the right of the bar
            i,             # Y coordinate
            f"{val:.2f}",  # Label text
            va="center",
            fontsize=11,
            color="black",
            weight="bold"
        )

    plt.title("Median of Features", fontsize=16)
    plt.xlabel("Median", fontsize=12)
    plt.ylabel("")
    sns.despine()
    plt.tight_layout()
    plt.show()


def mode_of_categorical_features(df):
    print("\n===== PART B: MODE OF IMPORTANT CATEGORICAL FEATURES =====")

    categorical_features = [
        "preferred_field_of_study",
        "motivation_level",
        "has_job",
        "financial_support_from_family",
        "stress_frequency",
        "communication_skills",
        "teacher_competence_satisfaction",
        "use_of_university_resources",
        "importance_of_interest_in_subject",
        "most_influencing_factor"
    ]

    mode_list = []
    for col in categorical_features:
        mode_value = df[col].mode()[0]
        mode_list.append([col, mode_value])

    mode_df = pd.DataFrame(mode_list, columns=["Feature", "Most Common Value"])
    print(mode_df.to_string(index=False))

    plt.figure(figsize=(20, 10))
    sns.barplot(
        data=mode_df,
        y="Feature",
        x="Most Common Value",
        hue="Most Common Value",
        dodge=False,
        palette="plasma",
        legend=False
    )
    plt.title("Mode of Features", fontsize=14)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def missing_values_check(df):
    # Mapping dictionaries for numeric encoding
    self_study_hours_map = {
        "less than 1 hour": 0,
        "1-3 hours": 2,
        "more than 4 hours": 4,
        "do not prepare": 0
    }

    exam_preparation_hours_map = {
        "less than 1 hour": 0,
        "2–8 hours": 5,
        "do not prepare": 0
    }

    motivation_map = {
        "low": 0,
        "average": 1,
        "moderately important": 1,
        "high": 2,
        "important": 2,
        "hard to say": 1
    }

    stress_map = {
        "never": 0,
        "rarely": 1,
        "sometimes": 2,
        "often": 3
    }

    interest_importance_map = {
        "slightly important": 0,
        "moderately important": 1,
        "important": 2,
        "do not prepare": 0
    }

    teacher_satisfaction_map = {
        "not satisfied at all": 0,
        "rather dissatisfied": 1,
        "hard to say": 1,
        "rather satisfied": 2
    }

    resources_use_map = {
        "never": 0,
        "sometimes": 1,
        "often": 2,
        "do not prepare": 0
    }

    family_support_map = {
        "no financial support": 0,
        "partial support": 1,
        "full support": 2
    }

    comm_skills_map = {
        "low": 0,
        "average": 1,
        "high": 2
    }

    # Apply mappings
    df['self_study_hours_num'] = df['self_study_hours'].map(self_study_hours_map)
    df['exam_preparation_hours_num'] = df['exam_preparation_hours'].map(exam_preparation_hours_map)
    df['motivation_num'] = df['motivation_level'].map(motivation_map)
    df['stress_num'] = df['stress_frequency'].map(stress_map)
    df['interest_importance_num'] = df['importance_of_interest_in_subject'].map(interest_importance_map)
    df['teacher_satisfaction_num'] = df['teacher_competence_satisfaction'].map(teacher_satisfaction_map)
    df['resources_use_num'] = df['use_of_university_resources'].map(resources_use_map)
    df['family_support_num'] = df['financial_support_from_family'].map(family_support_map)
    df['comm_skills_num'] = df['communication_skills'].map(comm_skills_map)

    print("\n===== PART C: MISSING VALUES CHECK =====")
    
    # Convert empty strings & whitespace to NaN
    df_check = df.replace(r'^\s*$', np.nan, regex=True)
    
    missing = df_check.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print(missing)
        plt.figure(figsize=(20, 10))
        sns.barplot(x=missing.values, y=missing.index, palette="magma")
        plt.title("Missing Values per Feature")
        plt.xlabel("Count")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values detected after cleaning.")



def outlier_analysis(df):
    print("\n===== PART D: OUTLIER ANALYSIS =====")
    df = df.replace({r'[–—‐-‒−]': '-'}, regex=True)

    # === Mapping dictionaries ===
    age_map = {
        "16-17": 16.5, "18-19": 18.5, "20-21": 20.5,
        "22-23": 22.5, "above 24": 25
    }

    gpa_map = {
        "below 2.7": 2.6, "2.8-3.0": 2.9, "3.1-3.2": 3.15,
        "3.3-3.4": 3.35, "above 3.5": 3.6
    }

    attendance_map = {
        "below 70%": 65, "70-79%": 74.5,
        "80-89%": 84.5, "90-100%": 95
    }

    study_map = {
        "do not study": 0, "less than 1 hour": 1,
        "1-3 hours": 2, "more than 4 hours": 3
    }

    sleep_map = {
        "less than 6 hours": 5, "7-9 hours": 8, "more than 10 hours": 11
    }

    prep_map = {
        "do not prepare": 0, "less than 1 hour": 1,
        "2-8 hours": 5, "9-15 hours": 12,
        "more than 16 hours": 16
    }

    # === Conversion ===
    df["age_num"] = df["age"].map(age_map)
    df["gpa_num"] = df["average_gpa"].map(gpa_map)
    df["study_num"] = df["self_study_hours"].map(study_map)
    df["sleep_num"] = df["average_sleep_hours"].map(sleep_map)
    df["prep_num"] = df["exam_preparation_hours"].map(prep_map)
    
    numeric_for_outliers = ["gpa_num", "study_num", "sleep_num", "prep_num"]

    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df[numeric_for_outliers], palette="viridis")
    plt.title("Outliers", fontsize=14)
    plt.ylabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    DATA = "/home/madiyar/work/homelab/master-thesis/data/processed_students_unlabeled.csv"
    plt.style.use("default")
    sns.set(style="whitegrid")
    df = pd.read_csv(DATA)
    cols_to_drop = DROP_COLUMNS_PROCESSED
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    range_cols = [
    "age",
    "average_gpa",
    "average_attendance",
    "self_study_hours",
    "average_sleep_hours",
    "exam_preparation_hours"
    ]

    for col in range_cols:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(parse_range_text)
    
    df = mapping(df)
            
    corr_features = [
        "age_num", "average_gpa_num", "average_attendance_num",
        "self_study_hours_num", "average_sleep_hours_num", "exam_preparation_hours_num",
        "motivation_num", "stress_num", "interest_importance_num",
        "teacher_satisfaction_num", "resources_use_num",
        "job_num", "family_support_num", "comm_skills_num", "gender_num"
    ]
    corr_features = [f for f in corr_features if f in df.columns]  
    corr_matrix = df[corr_features].corr(method="spearman")
        
    starting_point(df)
    
    # outlier_analysis(df)    
    # median_of_numerical_features(df)
    # mode_of_categorical_features(df)
    # student_and_gender(df)
    # age_of_students(df)
    # program_and_course(df)
    # correlation_heatmap(df, corr_matrix)
    sorted_correlation_heatmap(df, corr_matrix)
    
if __name__ == "__main__":
    main() 