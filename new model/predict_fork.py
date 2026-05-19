import pandas as pd
from catboost import CatBoostRegressor
from collections import Counter
import os
from dotenv import load_dotenv


load_dotenv()
# Прямые пути к файлам в папке data/
model_path = os.getenv("MODEL_OUTPUT_PATH")
db_path = os.getenv("DB_OUTPUT")

# Загружаем модель
model = CatBoostRegressor()
model.load_model(model_path)

# Загружаем базу вакансий для аналитики навыков
db = pd.read_csv(db_path, encoding='utf-8')

def get_fork_and_advices(input_data):
    df_custom = pd.DataFrame([input_data])
    
    # Модель возвращает двумерный массив: [[нижняя_граница, верхняя_граница]]
    prediction = model.predict(df_custom)
    pred_from = max(0, int(prediction[0][0]))
    pred_to = max(pred_from, int(prediction[0][1]))
    
    # Блок аналитики навыков
    current_skills = [s.strip().lower() for s in input_data['skills'].split(',')]
    
    # Ищем вакансии с такой же ролью/грейдом, но где средняя ЗП выше, чем предсказанный максимум пользователя
    rich_vacancies = db[
        (db['role'] == input_data['role']) & 
        (db['experience'] == input_data['experience']) & 
        (db['salary'] > pred_to)
    ]
    
    # Если в этой категории мало данных, расширяем поиск на грейд выше
    if len(rich_vacancies) < 5:
        rich_vacancies = db[db['experience'] == input_data['experience']]
        
    # Собираем все навыки высокооплачиваемых вакансий
    rich_skills_pool = []
    for skills_str in rich_vacancies['skills'].dropna():
        if skills_str != 'no_skills':
            rich_skills_pool.extend([s.strip().lower() for s in str(skills_str).split(',')])
            
    # Находим навыки, которых у нашего кандидата еще нет в стеке
    missing_skills = [skill for skill in rich_skills_pool if skill not in current_skills]
    
    # Берем топ-3 самых популярных недостающих навыков
    top_recommended = Counter(missing_skills).most_common(3)
    recommendations = [skill.capitalize() for skill, count in top_recommended if len(skill) > 1]
    
    return pred_from, pred_to, recommendations

# Тестовый запуск инференса
user_vacancy = {
    'name': 'Python Developer',
    'city': 'Москва',
    'experience': 'От 1 года до 3 лет',
    'schedule': 'Удаленная работа',
    'skills': 'Python, Git', 
    'role': 'Программист, разработчик',
    'industry': 'unknown',
    'employer': 'unknown',
}

salary_from, salary_to, skills_advice = get_fork_and_advices(user_vacancy)

print("==================================================")
print(f"Вакансия: {user_vacancy['name']}")
print(f"Рассчитанная вилка: от {salary_from:,} до {salary_to:,} руб.".replace(',', ' '))
print("==================================================")
print("РЕКОМЕНДАЦИЯ ПО СТЕКУ:")
if skills_advice:
    print("Чтобы пробить верхнюю планку и зарабатывать больше, добавьте в стек:")
    for idx, skill in enumerate(skills_advice, 1):
        print(f"  {idx}. {skill}")
else:
    print("Ваш стек соответствует топовым зарплатам!")
print("==================================================")
