from django.shortcuts import render
from drf_spectacular.utils import extend_schema, OpenApiResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .serializers import ResumeDataSerialazer, CalculateSalarySerialazer
@extend_schema(
    summary="Расчёт вилки дохода и рекомендации",
    description=(
        "Принимает данные резюме, возвращает прогноз зарплаты и список улучшений. В режиме заглушки"
    ),
    request=ResumeDataSerialazer,
    responses={
        200: CalculateSalarySerialazer,
        400: OpenApiResponse(description="Ошибка валидации входных данных")
    },
    tags=["Salary Prediction"]
)
@api_view(["POST"])
@parser_classes([JSONParser])
def calculate_salary(request):
    serializer = ResumeDataSerialazer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    data = serializer.data
    result = {
        "min_salary": 0,
        "max_salary": 100000,
        "recommendations": ['ПРОСНИСЬ и РАБОТАЙ', "спи больше"]
    }
    return Response(result, status=status.HTTP_200_OK)
@extend_schema(
    summary="Парсинг резюме файлика",
    description="Загружает файл резюме (PDF/DOCX/TXT, до 5 МБ), работает как заглушка пока",
    request={"multipart/form-data": {"type": "string", "format": "binary", "description": "Файл резюме"}},
    responses={
        200: ResumeDataSerialazer,
        400: OpenApiResponse(description="Ошибка: файл не передан, слишком большой или неверный формат")},
    tags=["Resume Parsing"]
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def parse_resume(request):
    result = {
        "role": "Python Backend Developer",
        "experience_years": 3,
        "skills": ["Python", "Django", "PostgreSQL", "Git", "Docker", "FastAPI"],
        "region": "Москва",
        "education": "Высшее техническое",
    }
    return Response(result , status=status.HTTP_200_OK)

