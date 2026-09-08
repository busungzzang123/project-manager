import base64
import json
import mimetypes
import os
import re
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()  # backend/.env 파일을 읽어서 환경변수로 등록

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

# 파일 검토 프롬프트에 넣을 추출 텍스트 최대 길이 (너무 길면 잘라서 보냄)
MAX_REVIEW_TEXT_CHARS = 12000


def _post_to_gemini(payload: dict, timeout: int = 30) -> dict:
    """Gemini API에 payload를 그대로 POST하고 JSON 응답을 반환합니다.
    에러 메시지에 API 키가 포함된 URL이 절대 노출되지 않도록 처리합니다.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다 (.env 확인 필요)")

    try:
        response = requests.post(
            GEMINI_API_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "알 수 없음"
        raise RuntimeError(f"Gemini API 호출 실패 (status={status})") from None
    except requests.exceptions.RequestException:
        raise RuntimeError("Gemini API 요청 중 네트워크 오류가 발생했습니다") from None

    return response.json()


def _extract_text_from_response(data: dict) -> str:
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("Gemini 응답 형식이 예상과 다릅니다") from None


def _call_gemini(prompt: str) -> str:
    """Gemini API에 프롬프트를 보내고, JSON 형식 응답 텍스트를 그대로 반환합니다.
    (계획 생성 / 역할 추천처럼 반드시 JSON으로만 답해야 하는 경우에 사용)
    """
    data = _post_to_gemini(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
    )
    return _extract_text_from_response(data)


def _extract_json(text: str):
    """혹시 ```json ... ``` 코드펜스로 감싸져 오면 벗겨내고 JSON으로 파싱합니다."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_plan_prompt(title: str, task_type: str, topic: str, deadline: str) -> str:
    today = date.today().isoformat()
    return f"""다음 과제 정보를 보고 세부 작업으로 나눠줘.

오늘 날짜: {today}
과제명: {title}
유형: {task_type}
주제: {topic}
최종 데드라인: {deadline}

유형 "{task_type}"의 특성에 맞는 세부 작업 흐름으로 나눠줘.
작업은 3~6개 정도가 적당해.

각 작업은 다음 정보를 포함해서 JSON 배열로만 답해줘 (다른 설명 텍스트는 절대 넣지 마):
- title: 작업 제목 (짧게)
- description: 작업에 대한 한두 문장 설명
- due_date: 오늘({today})보다 늦고 최종 데드라인({deadline})보다는 이르거나 같은 날짜로,
  작업 순서에 맞게 적절히 분배한 중간 마감일 (YYYY-MM-DD 형식). 절대 오늘보다 이전 날짜를 쓰지 마.
- review_criteria: 이 작업이 완료됐다고 볼 수 있는 구체적인 조건 (한두 문장)

예시 형식:
[
  {{"title": "...", "description": "...", "due_date": "2026-09-10", "review_criteria": "..."}}
]
"""


def build_role_prompt(tasks: list, members: list) -> str:
    task_lines = "\n".join(
        f"- (task_id={t['id']}) {t['title']}: {t.get('description') or '설명 없음'}"
        for t in tasks
    )
    member_lines = "\n".join(
        f"- {m['name']}: 강점=\"{m.get('strengths') or '없음'}\", "
        f"우선순위=\"{m.get('priority') or '없음'}\""
        for m in members
    )

    return f"""다음은 한 프로젝트의 작업 목록과 팀원 정보야.

작업 목록:
{task_lines}

팀원 정보:
{member_lines}

각 작업(task_id)마다 강점과 우선순위를 고려했을 때 가장 적합한 팀원 1명을 추천해줘.
가능하면 팀원들의 작업이 고르게 분배되도록 신경 써줘.

다음 JSON 배열로만 답해줘 (다른 설명 텍스트는 절대 넣지 마):
[
  {{"task_id": 숫자, "recommended_member": "팀원 이름", "reason": "추천 이유 한 줄"}}
]
"""


def build_review_prompt(task: dict, extracted_text: str | None) -> str:
    """업로드된 파일을 검토할 때 쓸 프롬프트. 작업(계획)의 정보를 기준으로 삼는다."""
    title = task.get("title", "")
    description = task.get("description") or "설명 없음"
    due_date = task.get("due_date") or "미정"
    review_criteria = task.get("review_criteria") or "명시된 완료 기준 없음"

    prompt = f"""너는 대학교 조별과제를 도와주는 꼼꼼하고 친절한 조교야.
아래는 한 팀원이 특정 작업(task)에 대해 제출한 결과물이야.
이 작업의 정보와 완료 기준을 기준으로 삼아서, 제출물이 기준을 충족하는지 검토하고
구체적이고 건설적인 피드백을 줘. 너무 냉정하거나 기죽이는 톤은 피하고,
잘한 점도 꼭 짚어주면서 실질적으로 도움이 되는 피드백을 줘.

[이 작업의 계획 정보]
제목: {title}
설명: {description}
마감일: {due_date}
완료 기준: {review_criteria}

피드백은 반드시 한국어 일반 텍스트로, 마크다운 기호(*, #, ``` 등) 없이 아래 형식을 지켜서 작성해줘:

1. 완료 기준 충족 여부: (충족 / 부분 충족 / 미충족) 중 하나를 쓰고, 한두 문장으로 이유 설명
2. 잘된 점: 1~2가지, 구체적으로
3. 보완이 필요한 점: 1~3가지, 구체적으로 (어떤 부분을 어떻게 고치면 좋을지)
4. 다음에 하면 좋을 일: 1~2가지
"""

    if extracted_text is not None:
        trimmed = extracted_text[:MAX_REVIEW_TEXT_CHARS]
        if len(extracted_text) > MAX_REVIEW_TEXT_CHARS:
            trimmed += "\n...(내용이 길어 일부만 검토했습니다)"
        prompt += f"\n[제출물 내용]\n{trimmed}\n"
    else:
        prompt += "\n[제출물]\n아래에 첨부된 이미지를 직접 보고 검토해줘.\n"

    return prompt


def _build_multimodal_parts(prompt: str, image_path: str | None):
    parts = [{"text": prompt}]
    if image_path:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})
    return parts


def recommend_roles(tasks: list, members: list):
    """LLM을 호출해서 작업별 추천 담당자를 생성합니다. 실패 시 예외를 던집니다."""
    if not tasks or not members:
        raise ValueError("작업 또는 팀원 목록이 비어 있습니다")

    prompt = build_role_prompt(tasks, members)
    raw_text = _call_gemini(prompt)
    recommendations = _extract_json(raw_text)

    if not isinstance(recommendations, list):
        raise ValueError("LLM 응답이 배열(JSON list) 형식이 아닙니다")

    return recommendations


def generate_task_plan(title: str, task_type: str, topic: str, deadline: str):
    """LLM을 호출해서 세부 작업 목록을 생성합니다. 실패 시 예외를 던집니다."""
    prompt = build_plan_prompt(title, task_type, topic, deadline)
    raw_text = _call_gemini(prompt)
    tasks = _extract_json(raw_text)

    if not isinstance(tasks, list):
        raise ValueError("LLM 응답이 배열(JSON list) 형식이 아닙니다")

    return tasks


def review_submission(task: dict, extracted_text: str | None = None, image_path: str | None = None) -> str:
    """업로드된 파일(텍스트 또는 이미지)을 작업 계획 기준으로 검토해서 피드백 텍스트를 반환합니다."""
    prompt = build_review_prompt(task, extracted_text)
    parts = _build_multimodal_parts(prompt, image_path)
    data = _post_to_gemini({"contents": [{"parts": parts}]}, timeout=60)
    return _extract_text_from_response(data)
