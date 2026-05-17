import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import ast
import os
from dotenv import load_dotenv


load_dotenv()
# Прямые пути к файлам
input_file = os.getenv("INPUT_DATASET_PATH")
model_output = os.getenv("MODEL_OUTPUT_PATH")
db_output = os.getenv("DB_OUTPUT")

print("Загружаем датасет для квантильной вилки из папки data/...")
df = pd.read_csv(input_file, sep=';', encoding='utf-8-sig', on_bad_lines='skip', engine='python')
df.columns = ['name', 'city', 'salary', 'experience', 'schedule', 'skills', 'role', 'industry', 'employer','grade']

# Очистка данных от нетехнических вакансий
garbage_patterns = 'продавец|консультант|станок|чпу|фрезеровщик|зубной|техник|художник|лаборант'
df = df[~df['name'].str.contains(garbage_patterns, case=False, na=False)]
df = df.dropna(subset=['salary', 'name'])
df['salary'] = df['salary'].astype(float)

def transform_skills(skills_val):
    if pd.isna(skills_val) or str(skills_val).strip() in ['', 'nan', 'NaN', '[]']: 
        return 'no_skills'
    try:
        parsed = ast.literal_eval(str(skills_val).strip())
        if isinstance(parsed, list): 
            return ", ".join(parsed)
    except: 
        pass
    return str(skills_val).replace('[', '').replace(']', '').replace("'", "").strip()

df['skills'] = df['skills'].apply(transform_skills)

# Подготовка признаков
y = df['salary']
features = ['name', 'city', 'experience', 'schedule', 'skills', 'role', 'industry', 'employer']
X = df[features].fillna('unknown')

cat_features = ['city', 'experience', 'schedule', 'role', 'industry', 'employer']
text_features = ['name', 'skills']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nОбучаем ОДНУ модель MultiQuantile для предсказания вилок...")
model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    text_features=text_features,
    loss_function='MultiQuantile:alpha=0.1,0.9',  # 0.1 — нижняя граница, 0.9 — верхняя граница
    random_seed=42,
    verbose=200
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Сохраняем готовую модель и базу в папку data/
model.save_model(model_output)
df[features + ['salary']].to_csv(db_output, index=False, encoding='utf-8')

print(f"\nМодель вилок успешно сохранена в файл: {model_output}")
