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
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]

    text = text.replace("'", '"')
    text = re.sub(r',(\s*[\}\]])', r'\1', text)
    
    parsed = json.loads(text)
    
    # Map possible key variations
    if isinstance(parsed, dict):
        if 'experience_years' in parsed and 'experience_year' not in parsed:
            parsed['experience_year'] = parsed['experience_years']
        if 'experience' in parsed and 'experience_year' not in parsed:
            parsed['experience_year'] = parsed['experience']
            
    return parsed


def parse_with_local_llm(raw_text: str):
    llm = get_llm()
    if not llm:
        raise RuntimeError("LLM model not found")

    prompt = f"""<|im_start|>system
You are a professional CV parser. Analyze the input resume and extract key details into a valid JSON object.
Return ONLY a JSON object. Do not include markdown codeblocks (do NOT use ```json).

JSON Schema:
{{
  "role": "string",
  "experience_year": integer,
  "skills": ["string"],
  "region": "string",
  "education": "string",
  "schedule": "string"
}}
<|im_end|>
<|im_start|>user
Resume:
{raw_text[:1200]}
<|im_end|>
<|im_start|>assistant
"""

def fallback_parse(raw_text: str) -> dict:
    parsed = {
        "role": "",
        "experience_year": 0,
        "skills": [],
        "region": "Москва",
        "education": "Нет",
        "schedule": "Полная занятость"
    }
    
    # Try to guess role from the first few lines
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    guessed_role = ""
    for line in lines[:8]:
        if any(keyword in line.lower() for keyword in ["developer", "разработчик", "engineer", "инженер", "аналитик", "тестировщик", "qa", "lead", "manager", "программист"]):
            if len(line) < 60:
                # Remove punctuation or bullets
                cleaned = re.sub(r'^[-\d\.\*\•\s]+', '', line).strip()
                guessed_role = cleaned
                break
    if not guessed_role and lines:
        guessed_role = lines[0][:50]
    parsed["role"] = guessed_role or "Разработчик"
    
    # Try to extract experience years
    exp_match = re.search(r'(\d+)\s*(?:лет|года|год|year|yr|опыт)', raw_text, re.IGNORECASE)
    if exp_match:
        parsed["experience_year"] = int(exp_match.group(1))
    else:
        # Fallback to check word experience
        if "нет опыта" in raw_text.lower() or "без опыта" in raw_text.lower():
            parsed["experience_year"] = 0
        
    # Extract skills by matching a predefined common list
    common_skills = [
        "python", "django", "flask", "fastapi", "git", "sql", "postgresql", "mysql", "sqlite",
        "docker", "kubernetes", "k8s", "aws", "linux", "javascript", "react", "vue", "angular",
        "html", "css", "c++", "c#", "java", "spring", "go", "golang", "redis", "celery", "rest api", "ci/cd"
    ]
    extracted_skills = []
    for skill in common_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if skill == "c++":
            pattern = r'c\+\+'
        if re.search(pattern, raw_text, re.IGNORECASE):
            cap_skill = skill.upper() if skill in ["sql", "html", "css", "aws", "redis", "git", "api", "k8s"] else skill.capitalize()
            extracted_skills.append(cap_skill)
    parsed["skills"] = extracted_skills
    
    # Try to guess region
    for city in ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", "Нижний Новгород", "Краснодар", "Сочи"]:
        if city.lower() in raw_text.lower():
            parsed["region"] = city
            break
            
    return parsed


def parse_with_local_llm(raw_text: str):
    llm = get_llm()
    if not llm:
        return fallback_parse(raw_text)

    prompt = f"""<|im_start|>system
You are a professional CV parser. Analyze the input resume and extract key details into a valid JSON object.
Return ONLY a JSON object. Do not include markdown codeblocks (do NOT use ```json).

JSON Schema:
{{
  "role": "string",
  "experience_year": integer,
  "skills": ["string"],
  "region": "string",
  "education": "string",
  "schedule": "string"
}}
<|im_end|>
<|im_start|>user
Resume:
{raw_text[:1200]}
<|im_end|>
<|im_start|>assistant
"""

    data = None
    try:
        output = llm(
            prompt,
            max_tokens=300,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False
        )
        raw_json = output["choices"][0]["text"].strip()
        data = clean_json_response(raw_json)
    except Exception as e:
        print(f"LLM parsing failed, using fallback parser. Error: {e}")
        data = fallback_parse(raw_text)

    # Ensure data is a dictionary
    if not isinstance(data, dict):
        data = fallback_parse(raw_text)

    # Guarantee all required/optional keys are present and have correct types
    if not data.get("role") or not str(data["role"]).strip():
        data["role"] = fallback_parse(raw_text).get("role", "Разработчик")
    else:
        data["role"] = str(data["role"]).strip()
        
    if not data.get("region") or not str(data["region"]).strip():
        data["region"] = "Москва"
    else:
        data["region"] = str(data["region"]).strip()
        
    if "skills" not in data or not isinstance(data["skills"], list):
        data["skills"] = []
    data["skills"] = [str(s).strip() for s in data["skills"] if str(s).strip()]
        
    if not data.get("education") or str(data["education"]).strip().lower() in ["", "none", "null", "undefined", "нет"]:
        data["education"] = "Нет"
    else:
        data["education"] = str(data["education"]).strip()
        
    if not data.get("schedule") or str(data["schedule"]).strip().lower() in ["", "none", "null", "undefined", "не указано"]:
        data["schedule"] = "Полная занятость"
    else:
        data["schedule"] = str(data["schedule"]).strip()

    # Safely convert experience_year to integer
    try:
        val = data.get("experience_year", 0)
        if val is None:
            data["experience_year"] = 0
        elif isinstance(val, str):
            digits = re.findall(r'\d+', val)
            if digits:
                data["experience_year"] = int(digits[0])
            else:
                data["experience_year"] = 0
        else:
            data["experience_year"] = int(val)
    except Exception:
        data["experience_year"] = 0

    return data


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