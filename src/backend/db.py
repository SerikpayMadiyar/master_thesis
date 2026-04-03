from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from models import Student, Teacher, Advisor
from sqlalchemy import select


DATABASE_URL = "postgresql+asyncpg://madiyar:YOUR_PASSWORD@localhost:5432/model_db"

engine = create_async_engine(DATABASE_URL, future=True, echo=False)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session

async def fetch_user_by_username(username: str):
    async with AsyncSessionLocal() as session:

        # Try student
        result = await session.execute(
            select(Student).where(Student.username == username)
        )
        student = result.scalar_one_or_none()

        if student:
            return {
                "role": "student",
                "id": student.student_id,
                "data": student
            }

        # Try teacher
        result = await session.execute(
            select(Teacher).where(Teacher.username == username)
        )
        teacher = result.scalar_one_or_none()

        if teacher:
            return {
                "role": "teacher",
                "id": teacher.teacher_id,
                "data": teacher
            }

        # Try advisor
        result = await session.execute(
            select(Advisor).where(Advisor.username == username)
        )
        advisor = result.scalar_one_or_none()

        if advisor:
            return {
                "role": "advisor",
                "id": advisor.advisor_id,
                "data": advisor
            }

        return None

async def fetch_students_by_program(educational_program: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Student).where(Student.educational_program == educational_program)
        )
        return result.scalars().all()


async def fetch_students_for_advisor(edu_program: str, course_year: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Student)
            .where(Student.educational_program == edu_program)
            .where(Student.year_of_study == f"{course_year} year")
        )
        return result.scalars().all()

async def fetch_student_by_id(student_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Student).where(Student.student_id == student_id)
        )
        return result.scalar_one_or_none()

async def fetch_teacher_by_id(teacher_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Teacher).where(Teacher.teacher_id == teacher_id)
        )
        return result.scalar_one_or_none()

async def fetch_advisor_by_id(advisor_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Advisor).where(Advisor.advisor_id == advisor_id)
        )
        return result.scalar_one_or_none()
