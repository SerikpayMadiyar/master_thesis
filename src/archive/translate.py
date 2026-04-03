import pandas as pd

translated_dict = {
    # 1
    "1. Ваш пол": "gender",
    "Мужской": "male",
    "Женский": "female",

    # 2
    "2. Ваш возраст": "age",
    "16-17": "16-17",
    "18-19": "18-19",
    "20-21": "20-21",
    "22-23": "22-23",
    "Больше 24": "above 24",

    # 3
    "3. Ваше образование": "degree",
    "Бакалавр": "bachelor",
    "Магистратура": "master",
    "Докторантура": "phd",

    # 4
    "4. Ваша образовательная программа": "educational_program",
    
    # 5
    "5. Ваш курс обучения": "year_of_study",
    "1 курс": "1 year",
    "2 курс": "2 year",
    "3 курс": "3 year",

    # 6
    "6. Средний балл (GPA) за последние два семестра (можно посмотреть во вкладке Transcript по ссылке https://du.astanait.edu.kz/transcript)?": "average_gpa",
    "Ниже 2.7": "below 2.7",
    "2.8-3.0": "2.8-3.0",
    "3.1-3.2": "3.1-3.2",
    "3.3-3.4": "3.3-3.4",
    "Выше 3.5": "above 3.5",

    # 7
    "7. Ваша средняя посещаемость предметов (можно посмотреть во вкладке Attendance -> All courses в lms.astanait.edu.kz)?": "average_attendance",
    "90-100%": "90-100%",
    "80-89%": "80-89%",
    "70-79%": "70-79%",
    "Ниже 70%": "below 70%",
    # 8
    "8. Предметы какой науки вам больше всего нравятся (например, история относится к гуманитарным наукам, математика к техническим)?": "preferred_field_of_study",
    "Гуманитарные науки": "humanities",
    "Технические науки": "technical sciences",
    "Компьютерные науки": "computer sciences",

    # 9
    "9. Сколько часов в день вы обычно уделяете самостоятельному обучению вне занятий?": "self_study_hours",
    "Не уделяю": "do not study",
    "Меньше часа": "less than 1 hour",
    "1-3 часа": "1-3 hours",
    "Больше 4 часов": "more than 4 hours",

    # 10
    "10. Сколько часов сна в среднем вы получаете за ночь?": "average_sleep_hours",
    "Меньше 6 часов": "less than 6 hours",
    "7-9 часов": "7-9 hours",
    "Больше 10 часов": "more than 10 hours",

    # 11
    "11. Как вы оцениваете свою мотивацию к учебе?": "motivation_level",
    "Очень высокая": "very high",
    "Высокая": "high",
    "Средняя": "average",
    "Низкая": "low",
    "Очень низкая": "very low",

    # 12
    "12. Есть ли у вас постоянная работа во время учебы?": "has_job",
    "Да, полный рабочий день": "yes, full-time",
    "Да, неполный рабочий день": "yes, part-time",
    "Нет": "no",

    # 13
    "13. Получаете ли вы финансовую поддержку от семьи во время учебы?": "financial_support_from_family",
    "Полная поддержка": "full support",
    "Частичная поддержка": "partial support",
    "Финансовой поддержки нет": "no financial support",

    # 14
    "14. Насколько часто вы испытываете стресс в связи с учебой?": "stress_frequency",
    "Очень часто": "very often",
    "Часто": "often",
    "Иногда": "sometimes",
    "Редко": "rarely",
    "Никогда": "never",

    # 15
    "15. Как вы оцениваете свои коммуникативные навыки (умение работать в группе, общаться с преподавателями и однокурсниками)?": "communication_skills",
    "Очень высоко": "very high",
    "Высоко": "high",
    "Средне": "average",
    "Низко": "low",
    "Очень низко": "very low",

    # 16
    "16. Насколько вы удовлетворены квалификацией и компетентностью ваших преподавателей?": "teacher_competence_satisfaction",
    "Полностью удовлетворен(а)": "fully satisfied",
    "Скорее удовлетворен(а)": "rather satisfied",
    "Трудно сказать": "hard to say",
    "Скорее не удовлетворен(а)": "rather dissatisfied",
    "Совсем не удовлетворен(а)": "not satisfied at all",

    # 17
    "17. Как часто вы пользуетесь университетскими ресурсами (библиотека, онлайн-курсы, лаборатории и т. д.)?": "use_of_university_resources",
    "Очень часто": "very often",
    "Часто": "often",
    "Иногда": "sometimes",
    "Редко": "rarely",
    "Никогда": "never",

    # 18
    "18. Насколько важен для вас интерес к предмету при изучении дисциплины?": "importance_of_interest_in_subject",
    "Очень важен": "very important",
    "Важен": "important",
    "Средне важен": "moderately important",
    "Слабо важен": "slightly important",
    "Не имеет значения": "not important",

    # 19
    "19. Сколько времени в среднем вы уделяете на подготовку к экзаменам?": "exam_preparation_hours",
    "Не уделяю": "do not prepare",
    "Менее 1 часа": "less than 1 hour",
    "2–8 часов": "2–8 hours",
    "9–15 часов": "9–15 hours",
    "Больше 16 часов": "more than 16 hours",

    # 20
    "20. Какие факторы, по вашему мнению, сильнее всего влияют на вашу академическую успеваемость?": "most_influencing_factor",
    "Личная мотивация": "personal motivation",
    "Поддержка семьи": "family support",
    "Финансовое положение": "financial situation",
    "Качество преподавания": "quality of teaching",
    "Интерес к предмету": "interest in the subject",
    "Объем самостоятельной работы": "amount of self-study",
    "Уровень стресса": "stress level"
}

# ---- LOAD YOUR EXCEL ----
df = pd.read_excel("survey_rus.xlsx")

# ---- TRANSLATE COLUMN NAMES ----
df.rename(columns=lambda col: translated_dict.get(col, col), inplace=True)

# ---- TRANSLATE EACH CELL ----
df = df.applymap(lambda x: translated_dict.get(x, x))

# ---- SAVE AS CSV ----
df.to_csv("survey_translated.csv", index=False)

print("Done! File saved as survey_translated.csv")
print(translated_dict.values())