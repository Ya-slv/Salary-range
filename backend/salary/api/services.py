import os
import json
import re
from llama_cpp import Llama
from docx import Document
# Путь к модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

_llm = None

def get_llm():
    global _llm
    if _llm is None and os.path.exists(MODEL_PATH): #Настройка
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=1024,
            n_threads=4,
            verbose=False,
            n_gpu_layers=0
        )
    return _llm


def clean_json_response(text: str):
    #Чинит ошибeи
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]

    text = text.replace("'", '"')
    text = re.sub(r',(\s*[\}\]])', r'\1', text)
    return json.loads(text)


def parse_with_local_llm(raw_text: str):
    llm = get_llm()
    if not llm:
        raise RuntimeError("LLM model not found")

    prompt = f"""Извлеки данные из резюме в формате JSON. Только ключи: role, experience_years, skills, region, education, schedule.
Пример ответа: {{"role": "Аналитик", "experience_years": 2, "skills": ["Atlassian", "Jira"], "region": "Санкт-Петербург", "education": "Высшее", schedule: "Полная занятость"}}
Текст: {raw_text[:1500]}"""

    output = llm(
        prompt,
        max_tokens=250,
        temperature=0,
        top_p=0.9,
        stop=["}", "</think>"],
        echo=False
    )
    raw_json = output["choices"][0]["text"].strip()
    return clean_json_response(raw_json)


def extract_text_from_file(file_obj) -> str:
    ext = file_obj.name.lower().split('.')[-1]
    file_obj.seek(0)

    if ext == 'pdf':
        import pdfplumber
        with pdfplumber.open(file_obj) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif ext == 'docx':
        doc = Document(file_obj)
        return "\n".join(para.text for para in doc.paragraphs)

    elif ext == 'doc':
        return file_obj.read().decode('utf-8', errors='ignore')

    else:
        return file_obj.read().decode('utf-8', errors='ignore')