from unittest.mock import MagicMock, patch
import pandas as pd
from api.ml_service import get_fork_and_advices, map_experience
from api.services import parse_with_local_llm, clean_json_response

def test_map_experience():
    assert map_experience(0) == 'Нет опыта'
    assert map_experience(2) == 'От 1 года до 3 лет'
    assert map_experience(5) == 'От 3 до 6 лет'
    assert map_experience(10) == 'Более 6 лет'

def test_clean_json_response():
    raw_json = "```json\n{\n  \"role\": \"Developer\"\n}\n```"
    cleaned = clean_json_response(raw_json)
    assert cleaned == {"role": "Developer"}

@patch("api.services.get_llm")
def test_parse_with_local_llm(mock_get_llm):
    mock_llm_instance = MagicMock()
    mock_get_llm.return_value = mock_llm_instance
    mock_llm_instance.return_value = {
        "choices": [
            {
                "text": '{"role": "Python Developer", "experience_years": 3, "skills": ["Python", "Git"], "region": "Москва", "education": "Высшее", "schedule": "Полная занятость"}'
            }
        ]
    }
    
    result = parse_with_local_llm("Resume text content")
    assert result["role"] == "Python Developer"
    assert result["experience_years"] == 3
    assert result["skills"] == ["Python", "Git"]

@patch("api.ml_service.os.path.exists")
@patch("api.ml_service.CatBoostRegressor")
@patch("api.ml_service.pd.read_csv")
def test_get_fork_and_advices_success(mock_read_csv, mock_catboost_cls, mock_exists):
    # Setup mocks for loading ML resources
    mock_exists.return_value = True
    
    # Mock CatBoost Model
    mock_model_instance = MagicMock()
    mock_catboost_cls.return_value = mock_model_instance
    mock_model_instance.predict.return_value = [[130000, 160000]]
    
    # Mock Pandas DataFrame for vacancies DB
    mock_df = pd.DataFrame([
        {"role": "Python Developer", "experience": "От 1 года до 3 лет", "salary": 170000, "skills": "Python, Django, Git, Docker"},
        {"role": "Python Developer", "experience": "От 1 года до 3 лет", "salary": 180000, "skills": "Python, Django, PostgreSQL"},
    ])
    mock_read_csv.return_value = mock_df
    
    # Reset global state to force reloading in tests
    import api.ml_service
    api.ml_service._model_loaded = False
    api.ml_service._db_loaded = False
    
    input_data = {
        "role": "Python Developer",
        "experience_year": 2,
        "skills": ["Python", "Django"],
        "region": "Москва",
        "schedule": "Полная занятость"
    }
    
    result = get_fork_and_advices(input_data)
    
    assert result["min_salary"] == 130000
    assert result["max_salary"] == 160000
    # Missing skills recommendation from mock_df (Git, Docker, PostgreSQL)
    assert len(result["recommendations"]) > 0
    
    # Convert all recommendations to lowercase for comparison
    recs_lower = [r.lower() for r in result["recommendations"]]
    assert any("git" in r or "docker" in r or "postgresql" in r for r in recs_lower)

@patch("api.ml_service.os.path.exists")
def test_get_fork_and_advices_fallback(mock_exists):
    mock_exists.return_value = False
    
    # Reset global state
    import api.ml_service
    api.ml_service._model_loaded = False
    api.ml_service._db_loaded = False
    api.ml_service._model = None
    api.ml_service._db = None
    
    input_data = {
        "role": "Python Developer",
        "experience_year": 2,
        "skills": ["Python"],
        "region": "Москва",
        "schedule": "Полная занятость"
    }
    
    result = get_fork_and_advices(input_data)
    assert result["min_salary"] == 50000
    assert result["max_salary"] == 150000
    assert any("модель" in r for r in result["recommendations"])
