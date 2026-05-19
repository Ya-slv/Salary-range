import pytest
from unittest.mock import patch, MagicMock
from rest_framework import status
from rest_framework.test import APIClient

@pytest.fixture
def api_client():
    return APIClient()

@patch("api.views.SalaryPredictionService")
def test_calculate_salary_success(mock_service_cls, api_client):
    # Mocking the service instance & predict return value
    mock_service_instance = MagicMock()
    mock_service_cls.return_value = mock_service_instance
    mock_service_instance.predict.return_value = {
        "min_salary": 120000,
        "max_salary": 140000,
        "recommendations": ["Excellent resume!"]
    }
    
    payload = {
        "role": "Python Backend Developer",
        "experience_year": 3,
        "skills": ["Python", "Django", "PostgreSQL", "Git", "Docker"],
        "region": "Москва",
        "education": "Высшее"
    }
    
    response = api_client.post("/api/calculate_salary/", payload, format="json")
    
    assert response.status_code == status.HTTP_200_OK
    assert response.data["min_salary"] == 120000
    assert response.data["max_salary"] == 140000
    assert response.data["recommendations"] == ["Excellent resume!"]
    mock_service_cls.assert_called_once()
    mock_service_instance.predict.assert_called_once()

@patch("api.views.SalaryPredictionService")
def test_calculate_salary_fallback(mock_service_cls, api_client):
    # Mock service to throw FileNotFoundError to test the fallback mechanism
    mock_service_instance = MagicMock()
    mock_service_cls.return_value = mock_service_instance
    mock_service_instance.predict.side_effect = FileNotFoundError("Model file not found")
    
    payload = {
        "role": "Python Backend Developer",
        "experience_year": 3,
        "skills": ["Python", "Django", "PostgreSQL"],
        "region": "Москва"
    }
    
    response = api_client.post("/api/calculate_salary/", payload, format="json")
    
    # Assert successful fallback response
    assert response.status_code == status.HTTP_200_OK
    
    # Expected base_salary = 50000 + 3 * 30000 + 3 * 5000 = 155000
    # min_salary = 155000 * 0.9 = 139500
    # max_salary = 155000 * 1.1 = 170500
    assert response.data["min_salary"] == 139500
    assert response.data["max_salary"] == 170500
    assert len(response.data["recommendations"]) > 0

def test_calculate_salary_invalid_data(api_client):
    # Missing required field 'role' and 'experience_year'
    payload = {
        "skills": ["Python"],
        "region": "Москва"
    }
    
    response = api_client.post("/api/calculate_salary/", payload, format="json")
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "role" in response.data
    assert "experience_year" in response.data
