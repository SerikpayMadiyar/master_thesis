from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RAW_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/output")
OUTPUT_FILE = OUTPUT_DIR / "processed_students_unlabeled.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS_RAW = ["Отметка времени", "Баллы"]


TRANSLATED_DICT = {
    # -----------------------------
    # Gender
    # -----------------------------
    "Ваш пол": "gender",
    "Мужской": "male",
    "Женский": "female",
    # -----------------------------
    # Age
    # -----------------------------
    "Ваш возраст": "age",
    "16-17": "16-17",
    "18-19": "18-19",
    "20-21": "20-21",
    "22-23": "22-23",
    "Больше 24": "above 24",
    # -----------------------------
    # Degree
    # -----------------------------
    "Ваше образование": "degree",
    "Бакалавр": "bachelor",
    "Магистратура": "master",
    "Докторантура": "phd",
    # -----------------------------
    # Educational program
    # -----------------------------
    "Ваша образовательная программа": "educational_program",
    # -----------------------------
    # Year of study
    # -----------------------------
    "Ваш курс обучения": "year_of_study",
    "1 курс": "1 year",
    "2 курс": "2 year",
    "3 курс": "3 year",
    # -----------------------------
    # Average GPA
    # -----------------------------
    "Средний балл (GPA) за последние два семестра (можно посмотреть во вкладке Transcript по ссылке https://du.astanait.edu.kz/transcript)?": "average_gpa",
    "2.0-2.7": "2.0-2.7",
    "Ниже 2.7": "below 2.7",
    "2.8-3.0": "2.8-3.0",
    "2.8-3.4": "2.8-3.4",
    "3.1-3.2": "3.1-3.2",
    "3.3-3.4": "3.3-3.4",
    "Выше 3.5": "above 3.5",
    # -----------------------------
    # Attendance
    # -----------------------------
    "Ваша средняя посещаемость предметов (можно посмотреть во вкладке Attendance -> All courses в lms.astanait.edu.kz)?": "average_attendance",
    "90-100%": "90-100%",
    "80-89%": "80-89%",
    "70-79%": "70-79%",
    "Ниже 70%": "below 70%",
    "50-69%": "50-69%",
    # -----------------------------
    # Preferred field of study
    # -----------------------------
    "Предметы какой науки вам больше всего нравятся (например, история относится к гуманитарным наукам, математика к техническим)?": "preferred_field_of_study",
    "Гуманитарные науки": "humanities",
    "Технические науки": "technical sciences",
    "Компьютерные науки": "computer sciences",
    # -----------------------------
    # Self-study hours
    # -----------------------------
    "Сколько часов в день вы обычно уделяете самостоятельному обучению вне занятий?": "self_study_hours",
    "Не уделяю": "do not study",
    "Меньше часа": "less than 1 hour",
    "1-3 часа": "1-3 hours",
    "Больше 4 часов": "more than 4 hours",
    # -----------------------------
    # Average sleep hours
    # -----------------------------
    "Сколько часов сна в среднем вы получаете за ночь?": "average_sleep_hours",
    "Меньше 6 часов": "less than 6 hours",
    "7-9 часов": "7-9 hours",
    "Больше 10 часов": "more than 10 hours",
    # -----------------------------
    # Motivation level
    # -----------------------------
    "Как вы оцениваете свою мотивацию к учебе?": "motivation_level",
    "Очень высокая": "very high",
    "Высокая": "high",
    "Средняя": "average",
    "Низкая": "low",
    "Очень низкая": "very low",
    # -----------------------------
    # Job status
    # -----------------------------
    "Есть ли у вас постоянная работа во время учебы?": "has_job",
    "Да, полный рабочий день": "yes, full-time",
    "Да, неполный рабочий день": "yes, part-time",
    "Нет": "no",
    # -----------------------------
    # Family support
    # -----------------------------
    "Получаете ли вы финансовую поддержку от семьи во время учебы?": "financial_support_from_family",
    "Полная поддержка": "full support",
    "Частичная поддержка": "partial support",
    "Финансовой поддержки нет": "no financial support",
    # -----------------------------
    # Stress frequency
    # -----------------------------
    "Насколько часто вы испытываете стресс в связи с учебой?": "stress_frequency",
    "Очень часто": "very often",
    "Часто": "often",
    "Иногда": "sometimes",
    "Редко": "rarely",
    "Никогда": "never",
    # -----------------------------
    # Communication skills
    # -----------------------------
    "Как вы оцениваете свои коммуникативные навыки (умение работать в группе, общаться с преподавателями и однокурсниками)?": "communication_skills",
    "Очень высоко": "very high",
    "Высоко": "high",
    "Средне": "average",
    "Низко": "low",
    "Очень низко": "very low",
    # -----------------------------
    # Satisfaction with teacher competence
    # -----------------------------
    "Насколько вы удовлетворены квалификацией и компетентностью ваших преподавателей?": "teacher_competence_satisfaction",
    "Полностью удовлетворен(а)": "fully satisfied",
    "Скорее удовлетворен(а)": "rather satisfied",
    "Трудно сказать": "hard to say",
    "Скорее не удовлетворен(а)": "rather dissatisfied",
    "Совсем не удовлетворен(а)": "not satisfied at all",
    # -----------------------------
    # Use of university resources
    # -----------------------------
    "Как часто вы пользуетесь университетскими ресурсами (библиотека, онлайн-курсы, лаборатории и т. д.)?": "use_of_university_resources",
    "Очень часто": "very often",
    "Часто": "often",
    "Иногда": "sometimes",
    "Редко": "rarely",
    "Никогда": "never",
    # -----------------------------
    # Importance of interest in subject
    # -----------------------------
    "Насколько важен для вас интерес к предмету при изучении дисциплины?": "importance_of_interest_in_subject",
    "Очень важен": "very important",
    "Важен": "important",
    "Средне важен": "moderately important",
    "Слабо важен": "slightly important",
    "Не имеет значения": "not important",
    # -----------------------------
    # Exam preparation hours
    # -----------------------------
    "Сколько времени в среднем вы уделяете на подготовку к экзаменам?": "exam_preparation_hours",
    "Не уделяю": "do not prepare",
    "Менее 1 часа": "less than 1 hour",
    "Менее 5 часов": "less than 5 hours",
    "2–8 часов": "2–8 hours",
    "5–10 часов": "5–10 hours",
    "9–15 часов": "9–15 hours",
    "Больше 16 часов": "more than 16 hours",
    "11–20 часов": "11–20 hours",
    "Более 20 часов": "more than 20 hours",
    # Иногда в данных могут быть другие тире/дефисы
    "2-8 часов": "2–8 hours",
    "5-10 часов": "5–10 hours",
    "9-15 часов": "9–15 hours",
    # -----------------------------
    # Most influencing factor
    # -----------------------------
    "Какие факторы, по вашему мнению, сильнее всего влияют на вашу академическую успеваемость?": "most_influencing_factor",
    "Личная мотивация": "personal motivation",
    "Поддержка семьи": "family support",
    "Финансовое положение": "financial situation",
    "Качество преподавания": "quality of teaching",
    "Интерес к предмету": "interest in the subject",
    "Объем самостоятельной работы": "amount of self-study",
    "Уровень стресса": "stress level",
}


def normalize_cell(value: Any) -> Any:
    """
    Normalizes a single cell value:
    - strips spaces for strings
    - leaves other types unchanged
    """
    if isinstance(value, str):
        return value.strip()
    return value


def find_excel_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    files.extend(input_dir.glob("*.xlsx"))
    files.extend(input_dir.glob("*.xls"))
    return sorted(files)


def read_excel_file(filepath: Path) -> pd.DataFrame:
    return pd.read_excel(filepath)


def translate_dataframe(
    df: pd.DataFrame, translation_dict: dict[str, str]
) -> pd.DataFrame:
    """
    Translates:
    - column names
    - string values
    """
    translated = df.copy()

    translated.columns = [
        translation_dict.get(normalize_cell(col), normalize_cell(col))
        for col in translated.columns
    ]

    translated = translated.map(normalize_cell)
    translated = translated.map(lambda x: translation_dict.get(x, x))

    return translated


def drop_unnecessary_columns(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    existing_columns = [col for col in drop_columns if col in df.columns]
    return df.drop(columns=existing_columns)


def add_student_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result.insert(0, "student_id", range(1, len(result) + 1))
    result.insert(1, "role", "student")
    result.insert(2, "username", result["student_id"].apply(lambda x: f"student_{x}"))
    result.insert(3, "password", result["student_id"].apply(lambda x: f"student_{x}"))

    return result


def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercases:
    - column names
    - all string cell values
    """
    result = df.copy()
    result.columns = result.columns.str.strip().str.lower()

    result = result.map(lambda x: x.strip().lower() if isinstance(x, str) else x)
    return result


def merge_raw_excel_files(input_dir: Path) -> pd.DataFrame:
    excel_files = find_excel_files(input_dir)

    if not excel_files:
        raise ValueError(f"No Excel files found in: {input_dir}")

    frames = []
    for file in excel_files:
        df = read_excel_file(file)
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)
    return merged_df


def save_dataframe_to_csv(df: pd.DataFrame, output_file: Path) -> None:
    df.to_csv(output_file, index=False)


def collect_untranslated_columns(
    df: pd.DataFrame, translation_dict: dict[str, str]
) -> list[str]:
    untranslated_columns: list[str] = []

    for col in df.columns:
        normalized_col = normalize_cell(col)
        if isinstance(normalized_col, str):
            if (
                normalized_col not in translation_dict
                and normalized_col != translation_dict.get(normalized_col)
            ):
                untranslated_columns.append(normalized_col)

    return sorted(set(untranslated_columns))


def collect_untranslated_values(
    df: pd.DataFrame, translation_dict: dict[str, str]
) -> dict[str, list[str]]:
    """
    Finds string values that are not translated by dictionary.
    Useful for debugging new survey variants or typos.
    """
    untranslated_by_column: dict[str, list[str]] = {}

    for col in df.columns:
        unique_values = df[col].dropna().unique().tolist()
        unknown_values: list[str] = []

        for value in unique_values:
            normalized_value = normalize_cell(value)

            if not isinstance(normalized_value, str):
                continue

            if normalized_value not in translation_dict:
                unknown_values.append(normalized_value)

        if unknown_values:
            untranslated_by_column[str(col)] = sorted(set(unknown_values))

    return untranslated_by_column


def print_untranslated_debug_report(
    df: pd.DataFrame, translation_dict: dict[str, str]
) -> None:
    untranslated_columns = collect_untranslated_columns(df, translation_dict)
    untranslated_values = collect_untranslated_values(df, translation_dict)

    print("\n=== UNTRANSLATED COLUMNS ===")
    if untranslated_columns:
        for col in untranslated_columns:
            print(col)
    else:
        print("No untranslated columns found.")

    print("\n=== UNTRANSLATED VALUES BY COLUMN ===")
    if untranslated_values:
        for col, values in untranslated_values.items():
            print(f"\nCOLUMN: {col}")
            for value in values:
                print(f"  - {value}")
    else:
        print("No untranslated values found.")


def save_untranslated_debug_report(
    df: pd.DataFrame,
    translation_dict: dict[str, str],
    output_file: Path,
) -> None:
    untranslated_columns = collect_untranslated_columns(df, translation_dict)
    untranslated_values = collect_untranslated_values(df, translation_dict)

    lines: list[str] = []

    lines.append("=== UNTRANSLATED COLUMNS ===")
    if untranslated_columns:
        lines.extend(untranslated_columns)
    else:
        lines.append("No untranslated columns found.")

    lines.append("")
    lines.append("=== UNTRANSLATED VALUES BY COLUMN ===")
    if untranslated_values:
        for col, values in untranslated_values.items():
            lines.append("")
            lines.append(f"COLUMN: {col}")
            for value in values:
                lines.append(f"  - {value}")
    else:
        lines.append("No untranslated values found.")

    output_file.write_text("\n".join(lines), encoding="utf-8")


def append_unique_values_from_processed_csv(
    csv_file: Path,
    output_file: Path,
    exclude_columns: list[str] | None = None,
) -> None:
    """
    Reads the final processed CSV and appends unique values
    for each column to the debug report file, excluding columns
    passed in exclude_columns.
    """
    if exclude_columns is None:
        exclude_columns = []

    df = pd.read_csv(csv_file)

    lines: list[str] = []
    lines.append("")
    lines.append("=== UNIQUE VALUES IN PROCESSED CSV ===")

    for col in df.columns:
        if col in exclude_columns:
            continue

        lines.append("")
        lines.append(f"COLUMN: {col}")

        unique_values = (
            df[col].dropna().astype(str).map(lambda x: x.strip()).unique().tolist()
        )

        unique_values = sorted(set(unique_values))

        for value in unique_values:
            lines.append(f"  - {value}")

    with output_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def build_processed_unlabeled_dataset(
    input_dir: Path,
    translation_dict: dict[str, str],
    drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - raw merged dataframe
    - processed dataframe
    """
    raw_df = merge_raw_excel_files(input_dir)

    translated_df = translate_dataframe(raw_df, translation_dict)
    translated_df = drop_unnecessary_columns(translated_df, drop_columns)
    translated_df = add_student_technical_columns(translated_df)
    translated_df = standardize_text(translated_df)

    return raw_df, translated_df


def main() -> None:
    raw_df, processed_df = build_processed_unlabeled_dataset(
        input_dir=RAW_DIR,
        translation_dict=TRANSLATED_DICT,
        drop_columns=DROP_COLUMNS_RAW,
    )

    save_dataframe_to_csv(processed_df, OUTPUT_FILE)

    debug_report_file = OUTPUT_DIR / "untranslated_debug_report.txt"
    save_untranslated_debug_report(
        df=raw_df,
        translation_dict=TRANSLATED_DICT,
        output_file=debug_report_file,
    )

    append_unique_values_from_processed_csv(
        csv_file=OUTPUT_FILE,
        output_file=debug_report_file,
        exclude_columns=["student_id", "role", "username", "password"],
    )

    print(f"Processed unlabeled dataset saved to: {OUTPUT_FILE}")
    print(f"Untranslated debug report saved to: {debug_report_file}")

    print_untranslated_debug_report(
        df=raw_df,
        translation_dict=TRANSLATED_DICT,
    )


if __name__ == "__main__":
    main()
