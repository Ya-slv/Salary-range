import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# --- ШАГ 1: ЗАГРУЗКА НАСТРОЕК ИЗ .ENV ---
load_dotenv()

# Достаем путь к файлу.
data_path = os.getenv("DATASET_PATH")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Критическая ошибка: Файл не найден по пути '{data_path}'! Проверьте файл .env")

print(f"1/4 Загрузка объединенного датасета из источника: {data_path}...")
df = pd.read_csv(data_path, sep=";")

# Выделяем признаки (X) и таргет (y)
features = ['name', 'city', 'experience_years', 'employment', 'schedule','skills', 'role', 'industry', 'employer','grade']
X = df[features].copy()
y = df['salary']

# Заполняем текстовые пропуски заглушками
for col in X.columns:
    X[col] = X[col].fillna('Не указано').astype(str)

print(f"Успешно загружено {X.shape[0]} вакансий для обучения модели.")





# --- ШАГ 2: ВЫДЕЛЕНИЕ ТАРГЕТА И РАЗДЕЛЕНИЕ НА ВЫБОРКИ ---
print("2/4 Подготовка выборок для обучения...")



# Выделяем таргет (целевую переменную зарплаты)
y = df['salary']

# Заполняем текстовые пропуски заглушками, чтобы CatBoost не ругался на пустые ячейки (NaN)
for col in X.columns:
    X[col] = X[col].fillna('Не указано').astype(str)

# Список всех текстовых (категориальных) колонок. 
# Мы передадим его CatBoost, чтобы он сам превратил их в умные числа.

cat_features = ['name', 'city', 'experience_years', 'employment', 'schedule', 'role', 'industry', 'employer','grade']

text_features = ['name', 'skills']

# Делим датасет на Обучающий (80%) и Тестовый (20%)
# random_state=42 фиксирует случайность, чтобы при каждом запуске результаты были одинаковыми
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape[0]} строк")
print(f"Размер тестовой выборки: {X_test.shape[0]} строк")






# --- ШАГ 3: ИНИЦИАЛИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ CATBOOST ---
print("3/4 Настройка параметров и запуск градиентного бустинга...")

model = CatBoostRegressor(
    iterations=400,           # Количество деревьев (шагов обучения). Оптимально для старта.
    learning_rate=0.03,        # Скорость обучения (шаг алгоритма).
    l2_leaf_reg=3,
    depth=6,                  # Глубина решающих деревьев. 6 — это золотой стандарт.
    loss_function='RMSE',     # Метрика, которую минимизирует модель (среднеквадратичная ошибка).
    random_seed=42,           # Наша любимая пасхалка для фиксации случайности.
    verbose=50               # Модель будет выводить отчет в консоль каждые 100 итераций.
)

print("Процесс пошел. Модель изучает рынок IT-вакансий...")
# Запускаем обучение. CatBoost сам заглянет в список cat_features и закодирует текст.
# Параметр eval_set позволяет модели на ходу тестировать себя на скрытых данных (X_test).
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    text_features=text_features,
    eval_set=(X_test, y_test),
    use_best_model=True       # В конце зафиксируется итерация с наилучшим результатом
)

print("Обучение успешно завершено!")




# --- ШАГ 4: ОЦЕНКА КАЧЕСТВА И ТЕСТОВЫЙ ПРОГНОЗ ---
print("4/4 Оценка точности модели...")

# Заставляем модель сделать предсказания для тестовой выборки
y_pred = model.predict(X_test)

# Вычисляем метрики
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*30)
print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ CATBOOST ===")
print("="*30)
print(f"Средняя абсолютная ошибка (MAE): {round(mae, 2)} руб.")
print(f"Коэффициент детерминации (R2):   {round(r2, 4)}")
print("="*30)

# --- ПРОВЕРКА НА СЛУЧАЙНОМ ЖИВОМ ПРИМЕРЕ ---
# Выбираем случайный индекс из тестовой выборки
sample_idx = X_test.sample(1, random_state=42).index[0]

print("\n>>> ТЕСТОВЫЙ ПРИМЕР ИЗ БАЗЫ ДАННЫХ:")
print(f"  Должность: {X_test.loc[sample_idx, 'name']}")
print(f"  Город:     {X_test.loc[sample_idx, 'city']}")
print(f"  Опыт:      {X_test.loc[sample_idx, 'experience_years']}")
print(f"  График:    {X_test.loc[sample_idx, 'schedule']}")
print(f"  Компания:  {X_test.loc[sample_idx, 'employer']}")
print("-" * 30)
print(f"  РЕАЛЬНАЯ ЗАРПЛАТА ИЗ ОБЪЯВЛЕНИЯ: {int(y_test.loc[sample_idx])} руб.")
print(f"  ПРОГНОЗ НАШЕЙ ML-МОДЕЛИ:        {int(y_pred[X_test.index.get_loc(sample_idx)])} руб.")
print("="*30)

# Сохраняем готовую модель в файл, чтобы бэкендеры могли подключить её к API
model_filename = "data/catboost_salary_model.cbm"
model.save_model(model_filename)
print(f"\nМодель успешно сохранена в файл: {model_filename}")



