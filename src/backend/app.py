import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import joblib

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

from db import (
    fetch_user_by_username,
    fetch_student_by_id,
    fetch_teacher_by_id,
    fetch_advisor_by_id,
)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="replace_this_with_a_random_secret")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ROLE_DASHBOARD = {
    "student": "/student/{id}/dashboard",
    "teacher": "/teacher/{id}/dashboard",
    "advisor": "/advisor/{id}/dashboard",
}

model = joblib.load(
    "/home/madiyar/01_projects/master-thesis/src/data_science/model/student_risk_semi_supervised.pkl"
)


def student_to_model_input(student: dict) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "gender": student["gender"],
                "age": student["age"],
                "degree": student["degree"],
                "educational_program": student["educational_program"],
                "year_of_study": student["year_of_study"],
                "average_gpa": student["average_gpa"],
                "average_attendance": student["average_attendance"],
                "preferred_field_of_study": student["preferred_field_of_study"],
                "self_study_hours": student["self_study_hours"],
                "average_sleep_hours": student["average_sleep_hours"],
                "motivation_level": student["motivation_level"],
                "has_job": student["has_job"],
                "financial_support_from_family": student[
                    "financial_support_from_family"
                ],
                "stress_frequency": student["stress_frequency"],
                "communication_skills": student["communication_skills"],
                "teacher_competence_satisfaction": student[
                    "teacher_competence_satisfaction"
                ],
                "use_of_university_resources": student["use_of_university_resources"],
                "importance_of_interest_in_subject": student[
                    "importance_of_interest_in_subject"
                ],
                "exam_preparation_hours": student["exam_preparation_hours"],
                "most_influencing_factor": student["most_influencing_factor"],
            }
        ]
    )

    return df


def predict_risk_level(student: dict):
    df = student_to_model_input(student)
    prediction = model.predict(df)[0]
    return prediction


@app.get("/")
async def index(request: Request):
    username = request.session.get("user")
    role = request.session.get("role")
    entity_id = request.session.get("entity_id")

    # User already logged in
    if username and role and entity_id:
        # Validate session with DB
        if role == "student":
            db_record = await fetch_student_by_id(entity_id)
        elif role == "teacher":
            db_record = await fetch_teacher_by_id(entity_id)
        elif role == "advisor":
            db_record = await fetch_advisor_by_id(entity_id)
        else:
            db_record = None

        # If user exists -> redirect to dashboard with ID
        if db_record:
            dashboard_url = ROLE_DASHBOARD[role].format(id=entity_id)
            return RedirectResponse(dashboard_url, status_code=302)

        # If not valid, clean session
        request.session.clear()

    return RedirectResponse("/login")


@app.get("/login")
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_post(
    request: Request, username: str = Form(...), password: str = Form(...)
):
    user = await fetch_user_by_username(username)

    if not user or user["data"].password != password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect username or password"},
        )

    request.session["user"] = username
    request.session["role"] = user["role"]
    request.session["entity_id"] = user["id"]

    dashboard_url = ROLE_DASHBOARD[user["role"]].format(id=user["id"])

    return RedirectResponse(dashboard_url, status_code=302)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


@app.get("/student/{student_id}/dashboard")
async def student_dashboard(student_id: int, request: Request):

    if request.session.get("role") != "student":
        return RedirectResponse("/login")

    session_student_id = request.session.get("entity_id")

    if session_student_id != student_id:
        return RedirectResponse(f"/student/{session_student_id}/dashboard")

    student = await fetch_student_by_id(student_id)

    if not student:
        return templates.TemplateResponse(
            "student_dashboard.html",
            {
                "request": request,
                "student": None,
                "viewer_role": "student",
                "error": "Студент не найден",
            },
        )

    try:
        risk_level = predict_risk_level(student)
    except Exception as e:
        risk_level = None
        print("Prediction error:", e)

    return templates.TemplateResponse(
        "student_dashboard.html",
        {
            "request": request,
            "student": student,
            "viewer_role": "student",
            "risk_level": risk_level,
        },
    )


@app.get("/teacher/group/{group_name}")
def teacher_group(request: Request, group_name: str):
    if request.session.get("role") != "teacher":
        return RedirectResponse("/login")
    sids = TEACHER_GROUPS.get(group_name, [])
    members = [get_student_by_id(s) for s in sids]
    members = [m for m in members if m]
    return templates.TemplateResponse(
        "teacher_group.html",
        {"request": request, "group_name": group_name, "members": members},
    )


@app.get("/teacher/dashboard")
def teacher_dashboard(request: Request):
    if request.session.get("role") != "teacher":
        return RedirectResponse("/login")
    groups_info = []
    for gname, sids in TEACHER_GROUPS.items():
        members = [get_student_by_id(s) for s in sids]
        members = [m for m in members if m]
        count = len(members)
        high_risk = sum(1 for m in members if m.get("risk_level") == "Высокий")
        risk_pct = int(100 * high_risk / count) if count else 0
        groups_info.append(
            {"name": gname, "count": count, "risk_pct": risk_pct, "members": members}
        )
    return templates.TemplateResponse(
        "teacher_dashboard.html", {"request": request, "groups": groups_info}
    )


@app.get("/teacher/student/{student_id}")
def teacher_student_dashboard(request: Request, student_id: int):
    role = request.session.get("role")
    if role not in ("teacher", "advisor"):
        return RedirectResponse("/login")
    student = get_student_by_id(student_id)
    if not student:
        return templates.TemplateResponse(
            "teacher_student_dashboard.html",
            {
                "request": request,
                "student": None,
                "metrics": None,
                "trend": None,
                "error": "Студент не найден",
                "viewer_role": role,
                "latest_responses": None,
            },
        )
    metrics = {
        "gpa": float(student.get("gpa") or 0),
        "attendance": attendance_to_numeric(student.get("attendance")),
        "self_study_hours": selfstudy_to_numeric(student.get("self_study_hours")),
    }
    trend = {
        "gpa_trend": [max(0, metrics["gpa"] - 0.2 + i * 0.05) for i in range(6)],
        "attendance_trend": [
            max(0, metrics["attendance"] - 20 + i * 5) for i in range(6)
        ],
        "study_trend": [
            metrics["self_study_hours"] + (-1) ** i * 0.5 * i for i in range(6)
        ],
    }
    latest = get_latest_responses(student_id)

    # --- compute risks ---
    general = compute_general_risk(student)
    domains = compute_domain_risks(student, general["percent"])

    return templates.TemplateResponse(
        "teacher_student_dashboard.html",
        {
            "request": request,
            "student": student,
            "metrics": metrics,
            "trend": trend,
            "viewer_role": role,
            "latest_responses": latest,
            "general_risk": general,
            "domain_risks": domains,
        },
    )


@app.get("/teacher/form/{student_id}")
def teacher_form_get(request: Request, student_id: int):
    if request.session.get("role") not in ("teacher", "advisor"):
        return RedirectResponse("/login")
    student = get_student_by_id(student_id)
    return templates.TemplateResponse(
        "teacher_form.html", {"request": request, "student": student, "error": None}
    )


@app.post("/teacher/form/{student_id}")
def teacher_form_post(
    request: Request,
    student_id: int,
    education_level: str = Form(...),
    gpa_range: str = Form(...),
    attendance: str = Form(...),
    self_study_hours: str = Form(...),
    sleep_hours: str = Form(...),
    motivation: str = Form(...),
    job: str = Form(...),
    financial_support: str = Form(...),
    stress: str = Form(...),
    communication: str = Form(...),
    teacher_quality: str = Form(...),
    use_resources: str = Form(...),
    interest_importance: str = Form(...),
    exam_prep_hours: str = Form(...),
    factors: list[str] | None = Form(None),
):
    if request.session.get("role") not in ("teacher", "advisor"):
        return RedirectResponse("/login")
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_id": student_id,
        "education_level": education_level,
        "gpa_range": gpa_range,
        "attendance": attendance,
        "self_study_hours": self_study_hours,
        "sleep_hours": sleep_hours,
        "motivation": motivation,
        "job": job,
        "financial_support": financial_support,
        "stress": stress,
        "communication": communication,
        "teacher_quality": teacher_quality,
        "use_resources": use_resources,
        "interest_importance": interest_importance,
        "exam_prep_hours": exam_prep_hours,
        "factors": ";".join(factors) if factors else "",
    }
    df = pd.DataFrame([record])
    if os.path.exists(RESPONSES_CSV):
        df.to_csv(RESPONSES_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(RESPONSES_CSV, index=False)
    return RedirectResponse(f"/teacher/student/{student_id}", status_code=302)


@app.get("/advisor/dashboard")
def advisor_dashboard(request: Request):
    if request.session.get("role") != "advisor":
        return RedirectResponse("/login")

    groups_info = []
    for gname, sids in ADVISOR_FLOWS.items():
        members = [get_student_by_id(s) for s in sids]
        members = [m for m in members if m]
        count = len(members)
        high_risk = sum(1 for m in members if m.get("risk_level") == "High")
        risk_pct = int(100 * high_risk / count) if count else 0
        groups_info.append(
            {"name": gname, "count": count, "risk_pct": risk_pct, "members": members}
        )

    return templates.TemplateResponse(
        "advisor_dashboard.html", {"request": request, "groups": groups_info}
    )


@app.get("/advisor/group/{group_name}")
def advisor_group(request: Request, group_name: str):
    if request.session.get("role") != "advisor":
        return RedirectResponse("/login")

    sids = ADVISOR_FLOWS.get(group_name, [])
    members = [get_student_by_id(s) for s in sids]
    members = [m for m in members if m]

    return templates.TemplateResponse(
        "teacher_group.html",
        {"request": request, "group_name": group_name, "members": members},
    )


@app.get("/advisor/form/{student_id}")
def advisor_form_get(request: Request, student_id: int):
    if request.session.get("role") != "advisor":
        return RedirectResponse("/login")
    student = get_student_by_id(student_id)
    return templates.TemplateResponse(
        "advisor_form.html", {"request": request, "student": student, "error": None}
    )


@app.post("/advisor/form/{student_id}")
def advisor_form_post(
    request: Request,
    student_id: int,
    education_level: str = Form(...),
    gpa_range: str = Form(...),
    attendance: str = Form(...),
    self_study_hours: str = Form(...),
    sleep_hours: str = Form(...),
    motivation: str = Form(...),
    job: str = Form(...),
    financial_support: str = Form(...),
    stress: str = Form(...),
    communication: str = Form(...),
    teacher_quality: str = Form(...),
    use_resources: str = Form(...),
    interest_importance: str = Form(...),
    exam_prep_hours: str = Form(...),
    factors: list[str] | None = Form(None),
    disability: str = Form(...),
    family_complete: str = Form(...),
    many_children: str = Form(...),
    family_financial_status: str = Form(...),
    parents_employed: str = Form(...),
    registered_in_school_time: str = Form(...),
):
    if request.session.get("role") != "advisor":
        return RedirectResponse("/login")
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "student_id": student_id,
        "education_level": education_level,
        "gpa_range": gpa_range,
        "attendance": attendance,
        "self_study_hours": self_study_hours,
        "sleep_hours": sleep_hours,
        "motivation": motivation,
        "job": job,
        "financial_support": financial_support,
        "stress": stress,
        "communication": communication,
        "teacher_quality": teacher_quality,
        "use_resources": use_resources,
        "interest_importance": interest_importance,
        "exam_prep_hours": exam_prep_hours,
        "factors": ";".join(factors) if factors else "",
        "disability": disability,
        "family_complete": family_complete,
        "many_children": many_children,
        "family_financial_status": family_financial_status,
        "parents_employed": parents_employed,
        "registered_in_school_time": registered_in_school_time,
    }
    # df = pd.DataFrame([record])
    # if os.path.exists(RESPONSES_CSV):
    #     df.to_csv(RESPONSES_CSV, mode='a', header=False, index=False)
    # else:
    #     df.to_csv(RESPONSES_CSV, index=False)
    return RedirectResponse(f"/teacher/student/{student_id}", status_code=302)


@app.get("/report/student/{student_id}")
def report_student(request: Request, student_id: int):

    def attendance_to_numeric(att):
        """
        Converts various attendance formats to a numeric percentage (0–100).
        Handles values like:
        - '95%'
        - '90-100'
        - '80-89%'
        - 'Above 90%'
        - 'Below 50%'
        - numeric types
        """
        if att is None:
            return 0.0

        s = str(att).strip()

        # Case: contains a dash or range
        if "-" in s:
            parts = [p.strip("% ") for p in s.split("-") if p.strip()]
            try:
                nums = [float(p) for p in parts]
                return sum(nums) / len(nums)
            except ValueError:
                pass

        # Case: textual ranges
        if "above" in s.lower():
            return float(s.lower().replace("above", "").replace("%", "").strip() or 0)
        if "below" in s.lower():
            return float(s.lower().replace("below", "").replace("%", "").strip() or 0)

        # Case: single number or with %
        s = s.replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    def selfstudy_to_numeric(v):
        if "1-3" in str(v):
            return 2
        if "4" in str(v):
            return 4
        if "Less" in str(v):
            return 0.5
        return float(v or 0)

    def sleep_to_numeric(v):
        if "7-9" in str(v):
            return 8
        if "6" in str(v):
            return 6
        if "10" in str(v):
            return 10
        return 7

    def exam_prep_to_numeric(v):
        if "5" in str(v):
            return 5
        if "10" in str(v):
            return 10
        if "20" in str(v):
            return 20
        return 8

    role = request.session.get("role")
    if role not in ("teacher", "advisor"):
        return RedirectResponse("/login")

    student = get_student_by_id(student_id)
    if not student:
        return RedirectResponse(f"/teacher/dashboard")

    # Convert metrics
    gpa = float(student.get("gpa") or 0)
    attendance_pct = attendance_to_numeric(student.get("attendance"))
    self_study = selfstudy_to_numeric(student.get("self_study_hours"))
    sleep = sleep_to_numeric(student.get("sleeping_hours"))
    exam_prep = exam_prep_to_numeric(student.get("preparing_to_exam_hours"))

    latest = get_latest_responses(student_id)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleStyle",
            fontSize=18,
            leading=22,
            alignment=1,
            textColor=colors.HexColor("#004c97"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeaderStyle",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#004c97"),
            spaceAfter=6,
        )
    )
    styles.add(ParagraphStyle(name="BodyStyle", fontSize=10, leading=12))

    elements = []

    # Title
    elements.append(
        Paragraph(f"Student Report: {student.get('name')}", styles["TitleStyle"])
    )
    elements.append(Spacer(1, 8))

    # Student Info Table
    student_info = [
        ["ID", student.get("student_id")],
        ["Program", "6B03201 Digital Journalism"],
        ["Favorite Subject", student.get("subject")],
        ["Risk Level", student.get("risk_level")],
        ["GPA", f"{gpa:.2f}"],
        ["Attendance", f"{attendance_pct:.1f}%"],
        ["Self-study Hours", student.get("self_study_hours")],
        ["Sleeping Hours", student.get("sleep_hours")],
        ["Exam Prep Hours", student.get("exam_prep_hours")],
    ]

    info_table = Table(student_info, hAlign="LEFT", colWidths=[120, 340])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#004c97")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    elements.append(info_table)
    elements.append(Spacer(1, 12))

    # Histogram: Self-study / Sleep / Exam prep hours
    elements.append(Paragraph("Daily Habits Overview", styles["HeaderStyle"]))
    drawing_hist = Drawing(420, 180)
    bar = VerticalBarChart()
    bar.x = 50
    bar.y = 40
    bar.height = 100
    bar.width = 300
    bar.data = [[self_study, sleep, exam_prep]]
    bar.barWidth = 20
    bar.strokeColor = colors.white
    bar.valueAxis.valueMin = 0
    bar.valueAxis.valueMax = 10
    bar.valueAxis.valueStep = 2
    bar.categoryAxis.categoryNames = ["Self-study", "Sleep", "Exam Prep"]
    bar.bars[0].fillColor = colors.HexColor("#004c97")
    drawing_hist.add(bar)
    drawing_hist.add(String(150, 150, "Hours per Day", fontSize=9))
    elements.append(drawing_hist)
    elements.append(Spacer(1, 20))

    # Attendance Pie Chart
    elements.append(Paragraph("Attendance Breakdown", styles["HeaderStyle"]))
    drawing_pie = Drawing(200, 180)
    pie = Pie()
    pie.x = 50
    pie.y = 20
    present = attendance_pct
    absent = max(0, 100 - present)
    pie.data = [present, absent]
    pie.labels = [f"Present {present:.1f}%", f"Absent {absent:.1f}%"]
    pie.slices.strokeWidth = 0.5
    pie.slices[0].fillColor = colors.HexColor("#1a75ff")
    pie.slices[1].fillColor = colors.HexColor("#ff6666")
    drawing_pie.add(pie)
    elements.append(drawing_pie)
    elements.append(Spacer(1, 20))

    # Survey Data
    elements.append(Paragraph("Information from Survey:", styles["HeaderStyle"]))
    survey_data = []
    for k, v in student.items():
        if k in ("student_id", "name", "subject", "risk_level", "gpa", "attendance"):
            continue
        survey_data.append([k.replace("_", " ").capitalize(), str(v)])
    if survey_data:
        survey_table = Table(survey_data, hAlign="LEFT", colWidths=[200, 250])
        survey_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        elements.append(survey_table)
        elements.append(Spacer(1, 20))

    # Teacher/Advisor survey section
    if latest:
        elements.append(
            Paragraph("Latest Teacher/Advisor Survey:", styles["HeaderStyle"])
        )
        latest_table = []
        for k, v in latest.items():
            if k in ("timestamp", "student_id"):
                continue
            latest_table.append(
                [k.replace("_", " ").capitalize(), str(v) if pd.notna(v) else "-"]
            )
        table = Table(latest_table, hAlign="LEFT", colWidths=[200, 250])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.beige),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 20))

    # Footer
    elements.append(
        Paragraph(
            f"<font size=8><i>Generated: {datetime.utcnow().isoformat()} | MVP System Report</i></font>",
            styles["BodyStyle"],
        )
    )

    doc.build(elements)
    buffer.seek(0)
    filename = f"report_student_{student_id}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
