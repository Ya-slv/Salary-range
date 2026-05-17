import json

from django.shortcuts import render
from drf_spectacular.utils import extend_schema, OpenApiResponse, OpenApiParameter
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .serializers import ResumeDataSerializer, CalculateSalarySerializer
from services import extract_text_from_file, parse_with_local_llm
@extend_schema(
    summary="Расчёт вилки дохода и рекомендации",
    description=(
        "Принимает данные резюме, возвращает прогноз зарплаты и список улучшений. В режиме заглушки"
    ),
    request=ResumeDataSerializer,
    responses={
        200: CalculateSalarySerializer,
        400: OpenApiResponse(description="Ошибка валидации входных данных")
    },
    tags=["Salary Prediction"]
)
@api_view(["POST"])
@parser_classes([JSONParser])
def calculate_salary(request):
    serializer = ResumeDataSerializer(data=request.data)
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
    summary="Парсинг резюме",
    description="Загружает файл резюме (PDF/DOCX/TXT, до 5 МБ) и извлекает данные через локальную LLM.",
    parameters=[
        OpenApiParameter(
            name="resume",
            type=OpenApiParameter.BINARY,
            location=OpenApiParameter.FORM,
            required=True,
            description="Файл резюме (PDF, DOCX, DOC, TXT)"
        )
    ],
    responses={
        200: ResumeDataSerializer,
        400: OpenApiResponse(description="Ошибка: файл не валиден или не удалось распарсить"),
        500: OpenApiResponse(description="Внутренняя ошибка сервера или модель не найдена")
    },
    tags=["Resume Parsing"]
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def parse_resume(request):
    resume_file = request.FILES.get('resume')
    if not resume_file:
        return Response({"error" : "нет файла"}, status=status.HTTP_400_BAD_REQUEST)

    if resume_file.size > 5 * 1024 * 1024:
        return Response({"error" : "Файл слишком большой"}, status=status.HTTP_400_BAD_REQUEST)

    ext = resume_file.name.lower().split('.')[-1]
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        return Response(
            {"error": "Поддерживаются только форматы: PDF, DOCX, DOC, TXT"},
            status=status.HTTP_400_BAD_REQUEST
        )
    try:
        raw_text = extract_text_from_file(resume_file)
        if not raw_text or len(raw_text.strip()) < 20:
            return Response({"error" : "Не удалось извлечь текст из файла"}, status=status.HTTP_400_BAD_REQUEST)

        parsed_data = parse_with_local_llm(raw_text)

        serializer = ResumeDataSerializer(data=parsed_data)
        if serializer.is_valid():
            return Response(serializer.validated_data, status=status.HTTP_200_OK)
        else:
            return Response(
                {"error": "Не удалось распарсить резюме: данные не соответствуют ожидаемому формату",
                "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST)
    except FileNotFoundError:
        return Response(
            {"error": "Файл модели не найден. Положите .gguf в backend/api/models/"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except RuntimeError as e:
        return Response(
        {"error": f"Ошибка инициализации модели: {str(e)}"},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except json.JSONDecodeError:
        return Response(
            {"error": "Не удалось распарсить резюме: модель вернула невалидный JSON"},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {"error": "Не удалось распарсить резюме. Попробуйте другой файл или формат."},
            status=status.HTTP_400_BAD_REQUEST
        )

