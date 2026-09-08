import os
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from db import get_conn, init_db
from file_extract import (
    get_file_kind,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
)
from llm import generate_task_plan, recommend_roles, review_submission

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 데모용. 실제 배포 시엔 프론트 URL로 좁히는 걸 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# AI 피드백 기능이 지원하는 파일 종류
SUPPORTED_REVIEW_KINDS = {"pdf", "docx", "pptx", "image"}


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


class ProjectCreate(BaseModel):
    title: str
    type: str  # "보고서형/PPT", "앱·웹 프로젝트", "케이스 스터디", "자료정리형", 또는 자유 텍스트
    topic: str
    deadline: str  # "2026-09-30" 형식의 문자열로 받음


def generate_share_code() -> str:
    # 짧고 URL에 넣기 좋은 코드 (예: "a3f9c1e2")
    return secrets.token_hex(4)


@app.post("/api/projects")
def create_project(payload: ProjectCreate):
    share_code = generate_share_code()

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO projects (title, type, topic, deadline, share_code) VALUES (?, ?, ?, ?, ?)",
            (payload.title, payload.type, payload.topic, payload.deadline, share_code),
        )
        conn.commit()
        project_id = cur.lastrowid

    return {
        "id": project_id,
        "title": payload.title,
        "type": payload.type,
        "topic": payload.topic,
        "deadline": payload.deadline,
        "share_code": share_code,
    }


@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, type, topic, deadline, share_code, created_at "
            "FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    return dict(row)


@app.get("/api/projects/by-code/{share_code}")
def get_project_by_code(share_code: str):
    """초대 링크(share_code)로 프로젝트를 조회합니다. 팀원이 참여할 때 씁니다."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, type, topic, deadline, share_code, created_at "
            "FROM projects WHERE share_code = ?",
            (share_code,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="유효하지 않은 초대 링크입니다")

    return dict(row)


# ---------------------------------------------------------------------------
# Day 2 — 작업(task) CRUD
# ---------------------------------------------------------------------------


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None
    review_criteria: Optional[str] = None


class TaskUpdate(BaseModel):
    done: Optional[bool] = None
    assignee_id: Optional[int] = None
    review_criteria: Optional[str] = None


def _project_exists(project_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    return row is not None


@app.post("/api/projects/{project_id}/tasks")
def create_task(project_id: int, payload: TaskCreate):
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO tasks (project_id, title, description, due_date, review_criteria)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, payload.title, payload.description, payload.due_date, payload.review_criteria),
        )
        conn.commit()
        task_id = cur.lastrowid

    return {"id": task_id, "project_id": project_id, "title": payload.title, "done": False}


@app.get("/api/projects/{project_id}/tasks")
def list_tasks(project_id: int):
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, title, description, due_date, review_criteria, done,
                   assignee_id, ai_recommended_assignee, ai_progress_estimate, ai_reasoning
            FROM tasks WHERE project_id = ? ORDER BY sort_order, id
            """,
            (project_id,),
        ).fetchall()

    tasks = [dict(row) for row in rows]
    for t in tasks:
        t["done"] = bool(t["done"])  # SQLite는 0/1로 저장하므로 bool로 변환

    total = len(tasks)
    done_count = sum(1 for t in tasks if t["done"])
    progress_percent = round((done_count / total) * 100, 1) if total > 0 else 0.0

    return {
        "progress_percent": progress_percent,
        "done_count": done_count,
        "total_count": total,
        "tasks": tasks,
    }


@app.patch("/api/tasks/{task_id}")
def update_task(task_id: int, payload: TaskUpdate):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

        fields = []
        values = []
        if payload.done is not None:
            fields.append("done = ?")
            values.append(1 if payload.done else 0)
        if payload.assignee_id is not None:
            fields.append("assignee_id = ?")
            values.append(payload.assignee_id)
        if payload.review_criteria is not None:
            fields.append("review_criteria = ?")
            values.append(payload.review_criteria)

        if fields:
            values.append(task_id)
            conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()

        updated = conn.execute(
            """
            SELECT id, title, description, due_date, review_criteria, done,
                   assignee_id, ai_recommended_assignee, ai_progress_estimate, ai_reasoning
            FROM tasks WHERE id = ?
            """,
            (task_id,),
        ).fetchone()

    result = dict(updated)
    result["done"] = bool(result["done"])
    return result


# ---------------------------------------------------------------------------
# Day 3 — LLM 계획 생성
# ---------------------------------------------------------------------------


@app.post("/api/projects/{project_id}/generate-plan")
def generate_plan(project_id: int):
    with get_conn() as conn:
        project = conn.execute(
            "SELECT id, title, type, topic, deadline FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

    if project is None:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    try:
        tasks_data = generate_task_plan(
            project["title"], project["type"], project["topic"], project["deadline"]
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI 계획 생성 실패: {exc}")

    created_ids = []
    with get_conn() as conn:
        for i, t in enumerate(tasks_data):
            cur = conn.execute(
                """
                INSERT INTO tasks (project_id, title, description, due_date, review_criteria, sort_order)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    t.get("title", "제목 없음"),
                    t.get("description"),
                    t.get("due_date"),
                    t.get("review_criteria"),
                    i,
                ),
            )
            created_ids.append(cur.lastrowid)
        conn.commit()

    return {"created_task_ids": created_ids, "count": len(created_ids)}


# ---------------------------------------------------------------------------
# Day 4 — 팀원 등록 + 역할 추천
# ---------------------------------------------------------------------------


class MemberCreate(BaseModel):
    name: str
    strengths: Optional[str] = None
    priority: Optional[str] = None


@app.post("/api/projects/{project_id}/members")
def create_member(project_id: int, payload: MemberCreate):
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO members (project_id, name, strengths, priority) VALUES (?, ?, ?, ?)",
            (project_id, payload.name, payload.strengths, payload.priority),
        )
        conn.commit()
        member_id = cur.lastrowid

    return {
        "id": member_id,
        "project_id": project_id,
        "name": payload.name,
        "strengths": payload.strengths,
        "priority": payload.priority,
    }


@app.get("/api/projects/{project_id}/members")
def list_members(project_id: int):
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, strengths, priority FROM members WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()

    return [dict(row) for row in rows]


@app.post("/api/projects/{project_id}/recommend-roles")
def recommend_project_roles(project_id: int):
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

    with get_conn() as conn:
        task_rows = conn.execute(
            "SELECT id, title, description FROM tasks WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        member_rows = conn.execute(
            "SELECT id, name, strengths, priority FROM members WHERE project_id = ?",
            (project_id,),
        ).fetchall()

    tasks = [dict(row) for row in task_rows]
    members = [dict(row) for row in member_rows]

    if not tasks:
        raise HTTPException(status_code=400, detail="작업이 없습니다. 먼저 계획을 생성해주세요")
    if not members:
        raise HTTPException(status_code=400, detail="등록된 팀원이 없습니다. 먼저 팀원을 등록해주세요")

    try:
        recommendations = recommend_roles(tasks, members)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI 역할 추천 실패: {exc}")

    valid_task_ids = {t["id"] for t in tasks}
    updated_count = 0

    with get_conn() as conn:
        for rec in recommendations:
            task_id = rec.get("task_id")
            recommended_member = rec.get("recommended_member")
            if task_id not in valid_task_ids or not recommended_member:
                continue  # 이상한 응답은 조용히 건너뜀 (사람이 나중에 직접 지정 가능)

            conn.execute(
                "UPDATE tasks SET ai_recommended_assignee = ?, ai_reasoning = ? WHERE id = ?",
                (recommended_member, rec.get("reason"), task_id),
            )
            updated_count += 1
        conn.commit()

    return {"updated_count": updated_count, "recommendations": recommendations}


# ---------------------------------------------------------------------------
# 파일 업로드/다운로드
# ---------------------------------------------------------------------------


@app.post("/api/tasks/{task_id}/files")
async def upload_task_file(
    task_id: int,
    file: UploadFile = File(...),
    uploader_name: str = Form(None),
):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    ext = os.path.splitext(file.filename)[1]
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO task_files (task_id, original_filename, stored_filename, uploader_name)
            VALUES (?, ?, ?, ?)
            """,
            (task_id, file.filename, stored_filename, uploader_name),
        )
        conn.commit()
        file_id = cur.lastrowid

    return {
        "id": file_id,
        "task_id": task_id,
        "original_filename": file.filename,
        "uploader_name": uploader_name,
    }


@app.get("/api/tasks/{task_id}/files")
def list_task_files(task_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, original_filename, uploader_name, uploaded_at, ai_feedback, ai_feedback_at
            FROM task_files WHERE task_id = ? ORDER BY uploaded_at DESC
            """,
            (task_id,),
        ).fetchall()

    return [dict(row) for row in rows]


@app.get("/api/files/{file_id}/download")
def download_file(file_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT original_filename, stored_filename FROM task_files WHERE id = ?",
            (file_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")

    file_path = os.path.join(UPLOAD_DIR, row["stored_filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일이 서버에 없습니다")

    return FileResponse(file_path, filename=row["original_filename"])


# ---------------------------------------------------------------------------
# AI 파일 피드백 — 업로드된 파일을 작업 계획 기준으로 검토
# ---------------------------------------------------------------------------


@app.post("/api/files/{file_id}/review")
def review_task_file(file_id: int):
    with get_conn() as conn:
        file_row = conn.execute(
            "SELECT id, task_id, original_filename, stored_filename FROM task_files WHERE id = ?",
            (file_id,),
        ).fetchone()
        if file_row is None:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")

        task_row = conn.execute(
            "SELECT id, title, description, due_date, review_criteria FROM tasks WHERE id = ?",
            (file_row["task_id"],),
        ).fetchone()
        if task_row is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    kind = get_file_kind(file_row["original_filename"])
    if kind not in SUPPORTED_REVIEW_KINDS:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 파일 형식입니다 (PDF, Word(.docx), PPT(.pptx), 이미지만 지원)",
        )

    file_path = os.path.join(UPLOAD_DIR, file_row["stored_filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일이 서버에 없습니다")

    extracted_text = None
    image_path = None

    try:
        if kind == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif kind == "docx":
            extracted_text = extract_text_from_docx(file_path)
        elif kind == "pptx":
            extracted_text = extract_text_from_pptx(file_path)
        elif kind == "image":
            image_path = file_path
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="파일 내용을 읽는 중 오류가 발생했습니다. 파일이 손상되지 않았는지 확인해주세요",
        ) from None

    if kind in ("pdf", "docx", "pptx") and not extracted_text:
        raise HTTPException(
            status_code=422,
            detail="파일에서 텍스트를 추출하지 못했습니다 (스캔 이미지로 된 PDF 등은 지원하지 않습니다)",
        )

    try:
        feedback = review_submission(dict(task_row), extracted_text=extracted_text, image_path=image_path)
    except Exception as exc:
        # llm.py에서 이미 API 키/URL이 안 남도록 정리된 메시지만 올라오므로 그대로 노출해도 안전함
        raise HTTPException(status_code=502, detail=f"AI 피드백 생성 실패: {exc}")

    generated_at = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE task_files SET ai_feedback = ?, ai_feedback_at = ? WHERE id = ?",
            (feedback, generated_at, file_id),
        )
        conn.commit()

    return {"id": file_id, "ai_feedback": feedback, "ai_feedback_at": generated_at}
