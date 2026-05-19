import json

from drf_spectacular.utils import extend_schema, OpenApiResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .serializers import ResumeDataSerializer, CalculateSalarySerializer
from .services import extract_text_from_file, parse_with_local_llm
from .ml_service import get_fork_and_advices

@extend_schema(
    summary="Расчёт вилки дохода и рекомендации",
    description=(
        "Принимает данные резюме, возвращает прогноз зарплаты и список улучшений."
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
    
    data = serializer.validated_data
    result = get_fork_and_advices(data)
    
    return Response(result, status=status.HTTP_200_OK)
@extend_schema(
    summary="Парсинг резюме",
    description="Загружает файл резюме (PDF/DOCX/TXT, до 5 МБ) и извлекает данные через локальную LLM.",
    request={
        'multipart/form-data': {
            'type': 'object',
            'properties': {
                'resume': {
                    'type': 'string',
                    'format': 'binary',
                    'description': 'Файл резюме (PDF, DOCX, DOC, TXT)'
                }
            },
            'required': ['resume']
        }
    },
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
    except FileNotFoundError as e:
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Файл модели не найден: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except RuntimeError as e:
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Ошибка модели: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except json.JSONDecodeError as e:
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Невалидный JSON от модели: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Не удалось распарсить резюме: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST
        )

