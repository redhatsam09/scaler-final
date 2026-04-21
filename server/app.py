import json
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from src.environment import DataCleaningEnv, DEFAULT_ENV_SEED
from src.models import Action
from src.graders import MissingValuesGrader, DuplicateHandlingGrader, ComplexValidationGrader

app = FastAPI(title="Data Cleaning Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

environments: Dict[str, DataCleaningEnv] = {}


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action: Dict[str, Any]


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    task_id: str
    step: int
    seed: int


class StepResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]


class GradeResponse(BaseModel):
    session_id: str
    task_id: str
    score: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "name": "Data Cleaning Environment",
        "version": "1.0.0",
        "tasks": ["task_missing_values", "task_duplicate_handling", "task_complex_validation"],
        "session_mode": "multi-session",
    }


def _coerce_seed(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_session_id(request: Request, payload: Dict[str, Any]) -> Optional[str]:
    query_session_id = request.query_params.get("session_id")
    if isinstance(query_session_id, str) and query_session_id.strip():
        return query_session_id.strip()

    header_session_id = request.headers.get("x-session-id")
    if isinstance(header_session_id, str) and header_session_id.strip():
        return header_session_id.strip()

    body_session_id = payload.get("session_id")
    if isinstance(body_session_id, str) and body_session_id.strip():
        return body_session_id.strip()

    return None


async def _extract_payload(request: Request) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    raw_body = await request.body()
    if not raw_body:
        return payload

    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        parsed_json = await request.json()
        if isinstance(parsed_json, dict):
            payload = parsed_json
    elif "application/x-www-form-urlencoded" in content_type:
        form_data = await request.form()
        payload = dict(form_data)
    else:
        text_body = raw_body.decode("utf-8", errors="ignore").strip()
        if text_body:
            try:
                parsed_text = json.loads(text_body)
                if isinstance(parsed_text, dict):
                    payload = parsed_text
            except json.JSONDecodeError:
                payload = {}
    return payload


def _resolve_runtime_session(session_id: Optional[str]) -> str:
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    if len(environments) == 1:
        return next(iter(environments))
    raise ValueError("Missing session_id. Provide session_id in body/query/header x-session-id.")


def _get_env_by_session(session_id: str) -> DataCleaningEnv:
    env = environments.get(session_id)
    if env is None:
        raise ValueError(f"Unknown session_id: {session_id}. Call /reset first.")
    return env


@app.post("/reset", response_model=ResetResponse)
@app.post("/reset/", response_model=ResetResponse)
async def reset(request: Request):
    try:
        payload = await _extract_payload(request)
        session_id = _extract_session_id(request, payload) or str(uuid.uuid4())

        task_id = request.query_params.get("task_id") or payload.get("task_id") or "task_missing_values"
        if not isinstance(task_id, str) or not task_id:
            task_id = "task_missing_values"

        seed = _coerce_seed(request.query_params.get("seed"))
        if seed is None:
            seed = _coerce_seed(payload.get("seed"))

        env = environments.get(session_id)
        if env is None:
            env = DataCleaningEnv(seed=seed if seed is not None else DEFAULT_ENV_SEED)
            environments[session_id] = env

        observation = env.reset(task_id=task_id, seed=seed)

        return ResetResponse(
            session_id=session_id,
            observation={
                "dataset_shape": observation.dataset_shape,
                "column_names": observation.column_names,
                "data_types": observation.data_types,
                "missing_values": observation.missing_values,
                "current_state": observation.current_state,
                "task_id": observation.task_id,
                "step_count": observation.step_count,
                "episode_progress": observation.episode_progress,
            },
            task_id=observation.task_id,
            step=observation.step_count,
            seed=env.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    try:
        session_id = _resolve_runtime_session(request.session_id)
        env = _get_env_by_session(session_id)
        if env.current_episode is None:
            raise ValueError("Environment not initialized. Call /reset first.")

        action = Action(**request.action)
        observation, reward, done, info = env.step(action)

        return StepResponse(
            session_id=session_id,
            observation={
                "dataset_shape": observation.dataset_shape,
                "column_names": observation.column_names,
                "data_types": observation.data_types,
                "missing_values": observation.missing_values,
                "current_state": observation.current_state,
                "task_id": observation.task_id,
                "step_count": observation.step_count,
                "episode_progress": observation.episode_progress,
            },
            reward=reward.value,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/state", response_model=StateResponse)
async def state(session_id: Optional[str] = None):
    try:
        resolved_session_id = _resolve_runtime_session(session_id)
        env = _get_env_by_session(resolved_session_id)
        return StateResponse(session_id=resolved_session_id, state=env.state())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade", response_model=GradeResponse)
async def grade(task_id: str = "task_missing_values", session_id: Optional[str] = None):
    try:
        resolved_session_id = _resolve_runtime_session(session_id)
        env = _get_env_by_session(resolved_session_id)
        if env.current_episode is None or env.current_episode.task_id != task_id:
            raise ValueError("Episode not matching requested task")

        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
        }

        grader_class = graders.get(task_id)
        if not grader_class:
            raise ValueError(f"Unknown task: {task_id}")

        score = grader_class.grade(env.current_episode)
        return GradeResponse(session_id=resolved_session_id, task_id=task_id, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
