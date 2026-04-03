# models.py
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True)
    role = Column(String)
    username = Column(String, unique=True)
    password = Column(String)
    gender = Column(String)
    age = Column(String)
    degree = Column(String)
    educational_program = Column(String)
    year_of_study = Column(String)
    average_gpa = Column(String)
    average_attendance = Column(String)
    preferred_field_of_study = Column(String)
    self_study_hours = Column(String)
    average_sleep_hours = Column(String)
    motivation_level = Column(String)
    has_job = Column(String)
    financial_support_from_family = Column(String)
    stress_frequency = Column(String)
    communication_skills = Column(String)
    teacher_competence_satisfaction = Column(String)
    use_of_university_resources = Column(String)
    importance_of_interest_in_subject = Column(String)
    exam_preparation_hours = Column(String)
    most_influencing_factor = Column(String)


class Teacher(Base):
    __tablename__ = "teachers"

    teacher_id = Column(Integer, primary_key=True)
    role = Column(String)
    username = Column(String, unique=True)
    password = Column(String)
    group_id = Column(Integer)


class Advisor(Base):
    __tablename__ = "advisors"

    advisor_id = Column(Integer, primary_key=True)
    role = Column(String)
    username = Column(String, unique=True)
    password = Column(String)
    educational_program = Column(String)
    course_year = Column(Integer)
