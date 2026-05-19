import pytest
from unittest.mock import patch
from rest_framework import status
from rest_framework.test import APIClient
from django.core.files.uploadedfile import SimpleUploadedFile

@pytest.fixture
def api_client():
    return APIClient()

@patch("api.views.get_fork_and_advices")
def test_calculate_salary_success(mock_get_fork_and_advices, api_client):
    mock_get_fork_and_advices.return_value = {
        "min_salary": 120000,
        "max_salary": 140000,
        "recommendations": ["Excellent resume!"]
    }
    
    payload = {
        "role": "Python Backend Developer",
        "experience_year": 3,
        "skills": ["Python", "Django", "PostgreSQL", "Git", "Docker"],
        "region": "Москва",
        "education": "Высшее",
        "schedule": "Полная занятость"
    }
    
    response = api_client.post("/api/calculate_salary/", payload, format="json")
    
    assert response.status_code == status.HTTP_200_OK
    assert response.data["min_salary"] == 120000
    assert response.data["max_salary"] == 140000
    assert response.data["recommendations"] == ["Excellent resume!"]
    mock_get_fork_and_advices.assert_called_once_with({
        "role": "Python Backend Developer",
        "experience_year": 3,
        "skills": ["Python", "Django", "PostgreSQL", "Git", "Docker"],
        "region": "Москва",
        "education": "Высшее",
        "schedule": "Полная занятость"
    })

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

@patch("api.views.extract_text_from_file")
@patch("api.views.parse_with_local_llm")
def test_parse_resume_success(mock_parse_llm, mock_extract, api_client):
    mock_extract.return_value = "Extracted text content from resume file."
    mock_parse_llm.return_value = {
        "role": "Python Developer",
        "experience_year": 2,
        "skills": ["Python", "Git"],
        "region": "Москва",
        "education": "Высшее",
        "schedule": "Полная занятость"
    }
    
    resume_file = SimpleUploadedFile("resume.docx", b"dummy docx content", content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
    response = api_client.post("/api/parse_resume/", {"resume": resume_file}, format="multipart")
    
    assert response.status_code == status.HTTP_200_OK
    assert response.data["role"] == "Python Developer"
    assert response.data["experience_year"] == 2
    assert response.data["skills"] == ["Python", "Git"]
    mock_extract.assert_called_once()
    mock_parse_llm.assert_called_once_with("Extracted text content from resume file.")

def test_parse_resume_no_file(api_client):
    response = api_client.post("/api/parse_resume/", {}, format="multipart")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "error" in response.data
