import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from api.services import SalaryPredictionService

def test_generate_recommendations():
    service = SalaryPredictionService(model_path="dummy_path.cbm")
    
    # Case 1: No experience, few skills, no Docker/Git
    data = {
        "experience_year": 0,
        "skills": ["Python"],
        "role": "Developer",
        "region": "Москва"
    }
    recs = service.generate_recommendations(data)
    assert "Накопите больше практического опыта (хотя бы 1 год)." in recs
    assert "Изучите дополнительные технологии и добавьте их в резюме." in recs
    assert "Рекомендуется освоить базовые инструменты разработки (Git, Docker)." in recs

    # Case 2: Good experience, enough skills, contains Git
    data = {
        "experience_year": 2,
        "skills": ["Python", "Django", "Git", "PostgreSQL", "Linux"],
        "role": "Developer",
        "region": "Москва"
    }
    recs = service.generate_recommendations(data)
    assert len(recs) == 1
    assert "Ваше резюме выглядит отлично! Продолжайте в том же духе." in recs

@patch("api.services.os.path.exists")
@patch("api.services.CatBoostRegressor")
def test_predict_calls_model(mock_catboost_cls, mock_exists):
    # Setup mock
    mock_exists.return_value = True
    mock_model_instance = MagicMock()
    mock_catboost_cls.return_value = mock_model_instance
    mock_model_instance.predict.return_value = [150000.0]

    service = SalaryPredictionService(model_path="dummy_path.cbm")
    
    data = {
        "role": "Python Developer",
        "experience_year": 3,
        "skills": ["Python", "Django", "Docker", "Git", "SQL"],
        "region": "Москва"
    }
    
    result = service.predict(data)
    
    # Assertions
    mock_model_instance.load_model.assert_called_once_with(service.model_path)
    mock_model_instance.predict.assert_called_once()
    
    # Check that input to predict is a DataFrame with expected columns
    args, kwargs = mock_model_instance.predict.call_args
    called_df = args[0]
    assert isinstance(called_df, pd.DataFrame)
    assert called_df.loc[0, 'name'] == "Python Developer"
    assert called_df.loc[0, 'experience_years'] == "3"
    assert "Python, Django, Docker, Git, SQL" in called_df.loc[0, 'skills']
    
    # Verify calculated bounds (150000 * 0.9 = 135000, 150000 * 1.1 = 165000)
    assert result["min_salary"] == 135000
    assert result["max_salary"] == 165000
    assert "Ваше резюме выглядит отлично! Продолжайте в том же духе." in result["recommendations"]

@patch("api.services.os.path.exists")
def test_model_file_not_found_raises_error(mock_exists):
    mock_exists.return_value = False
    service = SalaryPredictionService(model_path="nonexistent_model.cbm")
    
    with pytest.raises(FileNotFoundError):
        _ = service.model
