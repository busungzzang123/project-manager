import json
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


def _call_gemini(prompt: str) -> str:
    """Gemini API에 프롬프트를 보내고 응답 텍스트를 그대로 반환합니다."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다 (.env 확인 필요)")

    response = requests.post(
        GEMINI_API_URL,
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            # JSON 형식으로만 답하도록 API 레벨에서 강제
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Gemini 응답 형식이 예상과 다릅니다: {data}") from exc


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
