import pandas as pd

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv("survey_translated.csv")

# -----------------------------
# 2. Remove first two columns
# -----------------------------
df = df.drop(columns=["Отметка времени", "Баллы"])

# -----------------------------
# 3. Insert new columns
# -----------------------------

# student_id (1..N)
df.insert(0, "student_id", range(1, len(df) + 1))

# role = "student"
df.insert(1, "role", "student")

# username = student_1, student_2 ...
df.insert(2, "username", df["student_id"].apply(lambda x: f"student_{x}"))

# password = student_1, student_2 ...
df.insert(3, "password", df["student_id"].apply(lambda x: f"student_{x}"))

# -----------------------------
# 4. Compute risk_level
# -----------------------------

def parse_gpa(gpa_str):
    """
    Converts '3.1-3.2' → 3.15
    'above 3.5' → 3.5
    'below 2.7' → 2.6
    """
    gpa_str = gpa_str.lower().strip()

    if "below" in gpa_str:
        return float(gpa_str.replace("below", "").strip()) - 0.1  # approx
    if "above" in gpa_str:
        return float(gpa_str.replace("above", "").strip())
    if "-" in gpa_str:
        a, b = gpa_str.split("-")
        return (float(a) + float(b)) / 2
    return float(gpa_str)


def parse_attendance(att_str):
    """
    Converts '80-89%' → 84.5
    '90-100%' → 95
    'below 70%' → 69
    """
    att_str = att_str.lower().strip()
    att_str = att_str.replace("%", "")

    if "below" in att_str:
        return float(att_str.replace("below", "").strip()) - 1
    if "-" in att_str:
        a, b = att_str.split("-")
        return (float(a) + float(b)) / 2
    return float(att_str)


# convert strings → numeric
df["numeric_gpa"] = df["average_gpa"].apply(parse_gpa)
df["numeric_attendance"] = df["average_attendance"].apply(parse_attendance)

# risk rules:
def risk(gpa, att):

    # HIGH: gpa < 2.7
    if gpa < 2.7:
        return "high"

    # HIGH: attendance < 70%
    if att < 70:
        return "high"

    # MEDIUM: gpa 2.8–3.0 AND attendance 70–79%
    if 2.8 <= gpa <= 3.0 and 70 <= att < 80:
        return "medium"

    # ELSE → LOW
    return "low"


# df["risk_level"] = df.apply(lambda row: risk(row["numeric_gpa"], row["numeric_attendance"]), axis=1)

# drop helper columns
df = df.drop(columns=["numeric_gpa", "numeric_attendance"])

# -----------------------------
# 5. Convert all columns + values to lowercase
# -----------------------------
df.columns = df.columns.str.lower()         # lowercase column names

# lowercase all cells (only for string values)
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# -----------------------------
# 6. Save result
# -----------------------------
df.to_csv("processed_students_raw.csv", index=False)

print("Processing complete. Saved to processed_students.csv")
