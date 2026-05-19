import os
import pandas as pd
from django.conf import settings
from catboost import CatBoostRegressor

class SalaryPredictionService:
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to the data directory in the project root
            model_path = getattr(settings, 'SALARY_MODEL_PATH', os.path.join(settings.BASE_DIR, '..', '..', 'data', 'catboost_salary_model.cbm'))
        self.model_path = os.path.abspath(model_path)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            self._model = CatBoostRegressor()
            self._model.load_model(self.model_path)
        return self._model

    def predict(self, data):
        """
        Accepts a dictionary with keys: role, experience_year, skills, region, education.
        Returns predicted salary: min_salary, max_salary, recommendations.
        """
        skills_str = ", ".join(data.get('skills', [])) if isinstance(data.get('skills'), list) else data.get('skills', '')
        
        df = pd.DataFrame([{
            'name': data.get('role', 'Не указано'),
            'city': data.get('region', 'Не указано'),
            'experience_years': str(data.get('experience_year', 0)),
            'employment': 'Не указано',
            'schedule': 'Не указано',
            'skills': skills_str,
            'role': data.get('role', 'Не указано'),
            'industry': 'Не указано',
            'employer': 'Не указано',
            'grade': 'Не указано'
        }])

        # Predict using the ML model
        predicted_price = float(self.model.predict(df)[0])
        
        # Calculate range
        min_salary = int(predicted_price * 0.9)
        max_salary = int(predicted_price * 1.1)
        
        # Generating recommendations
        recommendations = self.generate_recommendations(data)
        
        return {
            "min_salary": max(0, min_salary),
            "max_salary": max(0, max_salary),
            "recommendations": recommendations
        }

    def generate_recommendations(self, data):
        recommendations = []
        experience = data.get('experience_year', 0)
        skills = data.get('skills', [])
        
        if experience < 1:
            recommendations.append("Накопите больше практического опыта (хотя бы 1 год).")
        if len(skills) < 5:
            recommendations.append("Изучите дополнительные технологии и добавьте их в резюме.")
        if "Docker" not in skills and "Git" not in skills:
            recommendations.append("Рекомендуется освоить базовые инструменты разработки (Git, Docker).")
        
        if not recommendations:
            recommendations.append("Ваше резюме выглядит отлично! Продолжайте в том же духе.")
            
        return recommendations
