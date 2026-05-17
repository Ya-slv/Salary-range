import os
import pandas as pd
import logging
from catboost import CatBoostRegressor
from collections import Counter
from django.conf import settings

logger = logging.getLogger(__name__)

# Fallback paths if not defined in .env
# BASE_DIR is backend/salary, so parent.parent is Salary-range
PROJECT_ROOT = settings.BASE_DIR.parent.parent
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'model.cbm')
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'vacancies.csv')

_model = None
_db = None
_model_loaded = False
_db_loaded = False

def load_ml_resources():
    global _model, _db, _model_loaded, _db_loaded
    
    model_path = os.getenv("MODEL_OUTPUT_PATH", DEFAULT_MODEL_PATH)
    db_path = os.getenv("DB_OUTPUT", DEFAULT_DB_PATH)
    
    if not _model_loaded:
        if os.path.exists(model_path):
            try:
                _model = CatBoostRegressor()
                _model.load_model(model_path)
                _model_loaded = True
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
        else:
            logger.warning(f"ML model file not found at {model_path}")
            
    if not _db_loaded:
        if os.path.exists(db_path):
            try:
                _db = pd.read_csv(db_path, encoding='utf-8', sep=None, engine='python')
                _db_loaded = True
                logger.info(f"Loaded vacancies DB from {db_path}")
            except Exception as e:
                logger.error(f"Failed to load DB: {e}")
        else:
            logger.warning(f"Vacancies DB not found at {db_path}")

def map_experience(experience_years):
    if experience_years == 0:
        return 'Нет опыта'
    elif 1 <= experience_years <= 3:
        return 'От 1 года до 3 лет'
    elif 3 < experience_years <= 6:
        return 'От 3 до 6 лет'
    else:
        return 'Более 6 лет'

def get_fork_and_advices(input_data):
    """
    input_data expected keys:
    - 'skills': list of strings
    - 'role': string
    - 'experience_year': integer
    - 'region': string
    - 'schedule': string
    """
    load_ml_resources()
    
    # Map input data to model format
    mapped_data = {
        'name': input_data.get('role', 'unknown'),
        'city': input_data.get('region', 'unknown'),
        'experience': map_experience(input_data.get('experience_year', 0)),
        'schedule': input_data.get('schedule', 'unknown'),
        'skills': ', '.join(input_data.get('skills', [])),
        'role': input_data.get('role', 'unknown'),
        'industry': 'unknown',
        'employer': 'unknown',
    }
    
    # If model or DB is missing, return fallback data
    if not _model_loaded or not _db_loaded:
        return {
            "min_salary": 50000,
            "max_salary": 150000,
            "recommendations": [
                "Добавьте модель .cbm в папку data/",
                "Добавьте базу вакансий .csv в папку data/"
            ]
        }
        
    df_custom = pd.DataFrame([mapped_data])
    
    try:
        prediction = _model.predict(df_custom)
        pred_from = max(0, int(prediction[0][0]))
        pred_to = max(pred_from, int(prediction[0][1]))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        pred_from, pred_to = 0, 100000

    recommendations = []
    if _db_loaded and _db is not None:
        try:
            current_skills = [s.strip().lower() for s in mapped_data['skills'].split(',')]
            
            # Check if salary column exists, some CSVs might have different names
            if 'salary' in _db.columns:
                rich_vacancies = _db[
                    (_db['role'] == mapped_data['role']) & 
                    (_db['experience'] == mapped_data['experience']) & 
                    (_db['salary'] > pred_to)
                ]
            else:
                rich_vacancies = pd.DataFrame()
            
            if len(rich_vacancies) < 5:
                rich_vacancies = _db[_db['experience'] == mapped_data['experience']]
                
            rich_skills_pool = []
            if 'skills' in rich_vacancies.columns:
                for skills_str in rich_vacancies['skills'].dropna():
                    if skills_str != 'no_skills':
                        rich_skills_pool.extend([s.strip().lower() for s in str(skills_str).split(',')])
                    
            missing_skills = [skill for skill in rich_skills_pool if skill not in current_skills]
            top_recommended = Counter(missing_skills).most_common(3)
            recommendations = [skill.capitalize() for skill, count in top_recommended if len(skill) > 1]
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            
    if not recommendations:
        recommendations = ["Ваш стек соответствует топовым зарплатам!"]

    return {
        "min_salary": pred_from,
        "max_salary": pred_to,
        "recommendations": recommendations
    }
