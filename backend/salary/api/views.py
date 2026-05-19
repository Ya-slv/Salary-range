from drf_spectacular.utils import extend_schema, OpenApiResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
import logging

from .serializers import ResumeDataSerialazer, CalculateSalarySerialazer
from .services import SalaryPredictionService

logger = logging.getLogger(__name__)

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
    
    try:
        service = SalaryPredictionService()
        result = service.predict(serializer.validated_data)
    except Exception as e:
        logger.warning(f"Failed to use ML model for salary prediction: {e}. Falling back to rule-based stub.")
        # Fallback to stub if model is not loaded/trained
        experience = serializer.validated_data.get('experience_year', 0)
        skills = serializer.validated_data.get('skills', [])
        
        # Simple rule-based mock prediction
        base_salary = 50000 + experience * 30000 + len(skills) * 5000
        min_salary = int(base_salary * 0.9)
        max_salary = int(base_salary * 1.1)
        
        # Generating recommendations inline to avoid using mocked service in fallback
        recommendations = []
        if experience < 1:
            recommendations.append("Накопите больше практического опыта (хотя бы 1 год).")
        if len(skills) < 5:
            recommendations.append("Изучите дополнительные технологии и добавьте их в резюме.")
        if "Docker" not in skills and "Git" not in skills:
            recommendations.append("Рекомендуется освоить базовые инструменты разработки (Git, Docker).")
        
        if not recommendations:
            recommendations.append("Ваше резюме выглядит отлично! Продолжайте в том же духе.")
        
        result = {
            "min_salary": min_salary,
            "max_salary": max_salary,
            "recommendations": recommendations
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

