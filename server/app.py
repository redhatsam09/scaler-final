import html
import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from src.environment import DataCleaningEnv, DEFAULT_ENV_SEED
from src.models import Action
from src.graders import (
    MissingValuesGrader,
    DuplicateHandlingGrader,
    ComplexValidationGrader,
    EnterpriseOrchestrationGrader,
)

SESSION_TTL_SECONDS = max(60, int(os.getenv("SESSION_TTL_SECONDS", "3600")))
MAX_ACTIVE_SESSIONS = max(1, int(os.getenv("MAX_ACTIVE_SESSIONS", "200")))
MAX_REASONING_LENGTH = max(128, int(os.getenv("MAX_REASONING_LENGTH", "4000")))

default_origins = ["https://huggingface.co"]
extra_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
ALLOWED_ORIGINS = list(dict.fromkeys(default_origins + extra_origins))

app = FastAPI(title="Enterprise Orchestration Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

environments: Dict[str, DataCleaningEnv] = {}
session_last_seen: Dict[str, float] = {}
session_locks: Dict[str, threading.Lock] = {}
registry_lock = threading.Lock()


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action: Dict[str, Any] = Field(default_factory=dict)


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


class CloseRequest(BaseModel):
    session_id: str


class CloseResponse(BaseModel):
    session_id: str
    closed: bool


@app.get("/health")
async def health():
    _cleanup_expired_sessions()
    with registry_lock:
        active_sessions = len(environments)
    return {
        "status": "ok",
        "active_sessions": active_sessions,
        "session_ttl_seconds": SESSION_TTL_SECONDS,
    }


@app.get("/")
async def root():
    _cleanup_expired_sessions()
    with registry_lock:
        active_sessions = len(environments)
    return {
        "name": "Enterprise Orchestration Environment",
        "version": "2.0.0",
        "tasks": [
            "task_missing_values",
            "task_duplicate_handling",
            "task_complex_validation",
            "task_enterprise_orchestration",
        ],
        "session_mode": "multi-session",
        "features": [
            "schema_drift", "actor_conflicts", "deceptive_oversight",
            "economic_budgets", "curriculum_difficulty", "stochastic_delegation",
            "natural_language_observations", "process_rewards",
        ],
        "session_ttl_seconds": SESSION_TTL_SECONDS,
        "max_active_sessions": MAX_ACTIVE_SESSIONS,
        "active_sessions": active_sessions,
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
    with registry_lock:
        if len(environments) == 1:
            return next(iter(environments))
    raise ValueError("Missing session_id. Provide session_id in body/query/header x-session-id.")


def _get_env_by_session(session_id: str) -> DataCleaningEnv:
    with registry_lock:
        env = environments.get(session_id)
        if env is not None:
            session_last_seen[session_id] = time.time()
    if env is None:
        raise KeyError(f"Unknown session_id: {session_id}. Call /reset first.")
    return env


def _get_session_lock(session_id: str) -> threading.Lock:
    with registry_lock:
        lock = session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            session_locks[session_id] = lock
        return lock


def _cleanup_expired_sessions() -> int:
    now = time.time()
    expired_ids: list[str] = []
    with registry_lock:
        for session_id, last_seen in list(session_last_seen.items()):
            if now - last_seen > SESSION_TTL_SECONDS:
                expired_ids.append(session_id)
        for session_id in expired_ids:
            environments.pop(session_id, None)
            session_last_seen.pop(session_id, None)
            session_locks.pop(session_id, None)
    return len(expired_ids)


def _create_or_get_env(session_id: str, seed: Optional[int]) -> DataCleaningEnv:
    with registry_lock:
        env = environments.get(session_id)
        if env is None:
            if len(environments) >= MAX_ACTIVE_SESSIONS:
                raise RuntimeError(
                    f"Maximum active sessions reached ({MAX_ACTIVE_SESSIONS}). Close idle sessions and retry."
                )
            env = DataCleaningEnv(seed=seed if seed is not None else DEFAULT_ENV_SEED)
            environments[session_id] = env
            session_locks[session_id] = threading.Lock()
        session_last_seen[session_id] = time.time()
        return env


def _close_session(session_id: str) -> bool:
    with registry_lock:
        existed = session_id in environments
        environments.pop(session_id, None)
        session_last_seen.pop(session_id, None)
        session_locks.pop(session_id, None)
    return existed


def _obs_to_dict(observation) -> Dict[str, Any]:
    return {
        "dataset_shape": observation.dataset_shape,
        "column_names": observation.column_names,
        "data_types": observation.data_types,
        "missing_values": observation.missing_values,
        "current_state": observation.current_state,
        "task_id": observation.task_id,
        "step_count": observation.step_count,
        "episode_progress": observation.episode_progress,
        "drift_notice": observation.drift_notice,
        "actor_messages": observation.actor_messages,
        "actor_objectives": observation.actor_objectives,
        "actor_conflicts": observation.actor_conflicts,
        "kpi_snapshot": observation.kpi_snapshot,
        "policy_version": observation.policy_version,
        "difficulty": observation.difficulty,
        "economic_status": observation.economic_status,
        "natural_language_observation": observation.natural_language_observation,
        "available_actions": observation.available_actions,
        "urgency_signals": observation.urgency_signals,
    }


def _compact_value(value: Any) -> str:
        if value is None:
                return "—"
        if isinstance(value, bool):
                return "Yes" if value else "No"
        if isinstance(value, int) and not isinstance(value, bool):
                return str(value)
        if isinstance(value, float):
                return f"{value:.3f}"
        if isinstance(value, (list, tuple, set, dict)):
                return json.dumps(value, ensure_ascii=False)
        return str(value)


def _mapping_rows(mapping: Dict[str, Any]) -> list[list[str]]:
        return [[str(key), _compact_value(value)] for key, value in mapping.items()]


def _render_html_table(title: str, rows: list[list[str]], headers: list[str]) -> str:
        if not rows:
                return f"""
                <section class=\"panel panel-soft\">
                    <div class=\"panel-title\">{html.escape(title)}</div>
                    <div class=\"panel-empty\">No data available.</div>
                </section>
                """

        header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
        body_rows = []
        for row in rows:
                cells = "".join(f"<td>{html.escape(_compact_value(cell))}</td>" for cell in row)
                body_rows.append(f"<tr>{cells}</tr>")

        return f"""
        <section class=\"panel panel-soft\">
            <div class=\"panel-title\">{html.escape(title)}</div>
            <div class=\"table-wrap\">
                <table class=\"dash-table\">
                    <thead><tr>{header_html}</tr></thead>
                    <tbody>{''.join(body_rows)}</tbody>
                </table>
            </div>
        </section>
        """


def _render_stat_card(label: str, value: Any, note: str = "", accent: str = "") -> str:
        accent_class = f" accent-{accent}" if accent else ""
        note_html = f"<div class=\"stat-note\">{html.escape(note)}</div>" if note else ""
        return f"""
        <div class=\"stat-card{accent_class}\">
            <div class=\"stat-label\">{html.escape(label)}</div>
            <div class=\"stat-value\">{html.escape(_compact_value(value))}</div>
            {note_html}
        </div>
        """


def _render_chip_list(items: list[str], empty_label: str = "None") -> str:
        if not items:
                return f"<span class=\"chip chip-muted\">{html.escape(empty_label)}</span>"
        return "".join(f"<span class=\"chip\">{html.escape(item)}</span>" for item in items)


def _render_dashboard_html(observation, task_id: str, difficulty: str, seed: str, reward: str, grade: str, done: bool, info: Dict[str, Any], history: list[Dict[str, Any]]) -> str:
        urgency_signals = list(observation.urgency_signals or [])
        actor_messages = list(observation.actor_messages or [])[-4:]
        objectives = list((observation.actor_objectives or {}).items())
        conflicts = list((observation.actor_conflicts or {}).items())
        available_actions = list(observation.available_actions or [])

        recent_history = history[-5:]
        history_items = [f"{entry['action']} ({entry['reward']:.3f})" for entry in recent_history]
        if not history_items:
                history_items = ["No steps executed yet. Use the controls on the left to initialize the episode."]

        status_label = "EPISODE COMPLETE" if done else "EPISODE ACTIVE"
        status_class = "status-done" if done else "status-live"
        drift_notice = observation.drift_notice or "No drift notice has been triggered yet."
        actor_cards = [
                f"<div class='mini-card'><span>Task</span><strong>{html.escape(task_id)}</strong></div>",
                f"<div class='mini-card'><span>Difficulty</span><strong>{html.escape(difficulty)}</strong></div>",
                f"<div class='mini-card'><span>Seed</span><strong>{html.escape(seed)}</strong></div>",
                f"<div class='mini-card'><span>Step</span><strong>{html.escape(str(observation.step_count))}</strong></div>",
        ]

        score_cards = [
                _render_stat_card("Step Reward", reward, "Current transition reward", "mint"),
                _render_stat_card("Grade", grade, "Task-specific grader score", "amber"),
                _render_stat_card("Policy Version", observation.policy_version, "Latest compliance rule set", "steel"),
                _render_stat_card("Budget Used", observation.economic_status.get("cost_used", "—"), f"Budget: {_compact_value(observation.economic_status.get('budget', '—'))}", "violet"),
        ]

        objective_rows = [[name, value] for name, value in objectives] or [["No actors", "No objectives available"]]
        conflict_rows = [[name, value] for name, value in conflicts] or [["No conflicts", "No conflict statements available"]]

        signal_cards = [
                _render_html_table("Actor Objectives", objective_rows, ["Actor", "Objective"]),
                _render_html_table("Conflict Map", conflict_rows, ["Pair", "Conflict"]),
        ]

        return f"""
        <section class=\"hero\">
            <div class=\"hero-copy\">
                <div class=\"eyebrow\">OpenEnv Hackathon 2026 · World Modeling × Multi-Agent Coordination</div>
                <h2>Enterprise Orchestration Console</h2>
                <p>A refined live demo for agentic reasoning under schema drift, budget pressure, and conflicting stakeholder incentives.</p>
            </div>
            <div class=\"hero-badges\">
                <div class=\"badge badge-primary\">+43.8% trained reward gain</div>
                <div class=\"badge\">0.831 held-out drift score</div>
                <div class=\"badge\">5 autonomous actors</div>
            </div>
        </section>

        <section class=\"panel panel-main\">
            <div class=\"status-row\">
                <span class=\"status-pill {status_class}\">{status_label}</span>
                <span class=\"status-pill\">Policy v{html.escape(str(observation.policy_version))}</span>
                <span class=\"status-pill\">{html.escape(str(observation.dataset_shape[0]))} rows × {html.escape(str(observation.dataset_shape[1]))} cols</span>
                <span class=\"status-pill\">{html.escape(_compact_value(observation.economic_status.get('cost_used', '—')))} used / {html.escape(_compact_value(observation.economic_status.get('budget', '—')))} budget</span>
            </div>

            <div class=\"stat-grid\">{''.join(actor_cards)}</div>

            <div class=\"panel-title\">Observation Narrative</div>
            <div class=\"narrative\">{html.escape(observation.natural_language_observation or 'No narrative observation is available yet.')}</div>

            <div class=\"panel-title\">Live Signals</div>
            <div class=\"signal-grid\">
                <div class=\"signal-card\">
                    <div class=\"signal-label\">Urgency</div>
                    <div class=\"signal-body\">{_render_chip_list(urgency_signals, 'No active alerts')}</div>
                </div>
                <div class=\"signal-card\">
                    <div class=\"signal-label\">Available Actions</div>
                    <div class=\"signal-body\">{_render_chip_list(available_actions, 'No actions reported')}</div>
                </div>
            </div>

            <div class=\"drift-box\">
                <div class=\"signal-label\">Drift / Policy Note</div>
                <div class=\"signal-body\">{html.escape(drift_notice)}</div>
            </div>

            <div class=\"two-col\">
                {''.join(score_cards)}
            </div>

            <div class=\"two-col\">
                {''.join(signal_cards)}
            </div>

            <div class=\"panel-title\">Recent Actor Messages</div>
            <ul class=\"bullet-list\">
                {''.join(f'<li>{html.escape(message)}</li>' for message in actor_messages) if actor_messages else '<li>No active communications</li>'}
            </ul>

            <div class=\"panel-title\">Execution History</div>
            <ul class=\"bullet-list\">
                {''.join(f'<li>{html.escape(item)}</li>' for item in history_items)}
            </ul>
        </section>
        """


def _suggest_next_steps(observation, history: list[Dict[str, Any]]) -> list[str]:
        suggestions: list[str] = []
        if not history:
                suggestions.append("Initialize with analyze to map the data quality before changing records.")
        if observation.drift_notice:
                suggestions.append("Use validate or oversight_review to adapt to drift before committing more changes.")
        if observation.urgency_signals:
                suggestions.append("Address the highest urgency signal first so the episode does not stall.")
        if observation.actor_messages:
                suggestions.append("Inspect the relevant actor before delegating if the request looks contradictory or risky.")
        if observation.missing_values:
                missing_total = sum(observation.missing_values.values())
                if missing_total > 0:
                        suggestions.append("Impute or validate the highest-missing columns to improve the quality index.")
        if "delegate" in observation.available_actions:
                suggestions.append("Delegate only after inspection when the actor incentives are materially different.")
        if not suggestions:
                suggestions.append("Continue with a disciplined analyze → act → validate loop.")
        return suggestions[:4]


@app.post("/reset", response_model=ResetResponse)
@app.post("/reset/", response_model=ResetResponse)
async def reset(request: Request):
    _cleanup_expired_sessions()
    try:
        payload = await _extract_payload(request)
        session_id = _extract_session_id(request, payload) or str(uuid.uuid4())

        task_id = request.query_params.get("task_id") or payload.get("task_id") or "task_missing_values"
        if not isinstance(task_id, str) or not task_id:
            task_id = "task_missing_values"
        difficulty = request.query_params.get("difficulty") or payload.get("difficulty")
        if difficulty is not None and not isinstance(difficulty, str):
            difficulty = None

        seed = _coerce_seed(request.query_params.get("seed"))
        if seed is None:
            seed = _coerce_seed(payload.get("seed"))

        env = _create_or_get_env(session_id, seed)
        session_lock = _get_session_lock(session_id)

        with session_lock:
            observation = env.reset(task_id=task_id, seed=seed, difficulty=difficulty)

        return ResetResponse(
            session_id=session_id,
            observation=_obs_to_dict(observation),
            task_id=observation.task_id,
            step=observation.step_count,
            seed=env.seed,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    _cleanup_expired_sessions()
    try:
        session_id = _resolve_runtime_session(request.session_id)
        action_payload = request.action if isinstance(request.action, dict) else {}
        reasoning_text = str(action_payload.get("reasoning", ""))
        if len(reasoning_text) > MAX_REASONING_LENGTH:
            raise ValueError(f"Reasoning exceeds max length ({MAX_REASONING_LENGTH} chars).")

        session_lock = _get_session_lock(session_id)
        with session_lock:
            env = _get_env_by_session(session_id)
            if env.current_episode is None:
                raise ValueError("Environment not initialized. Call /reset first.")

            action = Action(**action_payload)
            observation, reward, done, info = env.step(action)

        return StepResponse(
            session_id=session_id,
            observation=_obs_to_dict(observation),
            reward=reward.value,
            done=done,
            info=info
        )
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/state", response_model=StateResponse)
async def state(session_id: Optional[str] = None):
    _cleanup_expired_sessions()
    try:
        resolved_session_id = _resolve_runtime_session(session_id)
        session_lock = _get_session_lock(resolved_session_id)
        with session_lock:
            env = _get_env_by_session(resolved_session_id)
            return StateResponse(session_id=resolved_session_id, state=env.state())
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade", response_model=GradeResponse)
async def grade(task_id: str = "task_missing_values", session_id: Optional[str] = None):
    _cleanup_expired_sessions()
    try:
        resolved_session_id = _resolve_runtime_session(session_id)
        session_lock = _get_session_lock(resolved_session_id)
        with session_lock:
            env = _get_env_by_session(resolved_session_id)
            if env.current_episode is None:
                raise ValueError("Environment not initialized. Call /reset first.")
            if env.current_episode.task_id != task_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Task mismatch: active episode is '{env.current_episode.task_id}' "
                        f"but grade requested for '{task_id}'."
                    ),
                )

        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
            "task_enterprise_orchestration": EnterpriseOrchestrationGrader,
        }

        grader_class = graders.get(task_id)
        if not grader_class:
            raise ValueError(f"Unknown task: {task_id}")

        score = grader_class.grade(env.current_episode)
        return GradeResponse(session_id=resolved_session_id, task_id=task_id, score=score)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close", response_model=CloseResponse)
async def close_session(request: CloseRequest):
    _cleanup_expired_sessions()
    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    closed = _close_session(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return CloseResponse(session_id=session_id, closed=True)


# ---- Gradio Interactive Demo ----

def _build_gradio_demo():
    """Build Gradio UI for judges to interact with the environment."""
    try:
        import gradio as gr
        from pathlib import Path
    except ImportError:
        return None

    demo_env = DataCleaningEnv(seed=42)
    demo_session = {"obs": None, "history": [], "task_id": "task_enterprise_orchestration", "seed": 42, "difficulty": "medium"}

    def _empty_reward_rows() -> list[list[str]]:
        return [["Reward", "0.0000"], ["Grade", "0.0000"], ["Budget used", "—"], ["Budget remaining", "—"]]

    def _observation_tables(obs) -> tuple[list[list[str]], list[list[str]]]:
        schema_rows = [[column, obs.data_types.get(column, "—"), obs.missing_values.get(column, 0)] for column in obs.column_names]
        kpi_rows = [[metric, value] for metric, value in obs.kpi_snapshot.items()]
        if not kpi_rows:
            kpi_rows = [["No KPIs reported", "—"]]
        return schema_rows, kpi_rows

    def _build_reward_rows(obs, reward_value: float, grade_value: float, info: Dict[str, Any]) -> list[list[str]]:
        components = info.get("components", {}) if isinstance(info, dict) else {}
        budget = obs.economic_status.get("budget", None)
        used = obs.economic_status.get("cost_used", None)
        remaining = None
        if isinstance(budget, (int, float)) and isinstance(used, (int, float)):
            remaining = max(0.0, float(budget) - float(used))

        rows = [["Reward", f"{reward_value:.4f}"], ["Grade", f"{grade_value:.4f}"]]
        if budget is not None:
            rows.append(["Budget", _compact_value(budget)])
        if used is not None:
            rows.append(["Budget used", _compact_value(used)])
        if remaining is not None:
            rows.append(["Budget remaining", f"{remaining:.2f}"])
        for key, value in components.items():
            rows.append([f"Component: {key}", _compact_value(value)])
        return rows or _empty_reward_rows()

    def _build_raw_payload(obs, reward_value: float, grade_value: float, done: bool, info: Dict[str, Any], action_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": demo_session["task_id"],
            "observation": _obs_to_dict(obs),
            "reward": reward_value,
            "grade": grade_value,
            "done": done,
            "history": demo_session["history"],
            "action": action_payload,
            "info": info,
        }

    def reset_env(task_id, difficulty, seed):
        seed_val = int(seed) if seed else 42
        obs = demo_env.reset(task_id=task_id, seed=seed_val, difficulty=difficulty)
        demo_session["obs"] = obs
        demo_session["history"] = []
        demo_session["task_id"] = task_id
        demo_session["seed"] = seed_val
        demo_session["difficulty"] = difficulty

        schema_rows, kpi_rows = _observation_tables(obs)
        reward_rows = _empty_reward_rows()
        history_text = "No execution history recorded yet."
        guidance_text = "\n".join(f"- {step}" for step in _suggest_next_steps(obs, demo_session["history"]))
        raw_payload = _build_raw_payload(obs, 0.0, 0.0, False, {"components": {}}, {"event": "reset", "task_id": task_id, "difficulty": difficulty, "seed": seed_val})

        return (
            _render_dashboard_html(obs, task_id, difficulty, str(seed_val), "0.0000", "0.0000", False, {"components": {}}, demo_session["history"]),
            schema_rows,
            kpi_rows,
            reward_rows,
            history_text,
            guidance_text,
            raw_payload,
        )

    def step_env(action_type, target_cols, params_json, reasoning):
        if demo_session["obs"] is None:
            empty_obs = demo_env.reset(task_id=demo_session["task_id"], seed=42)
            schema_rows, kpi_rows = _observation_tables(empty_obs)
            return (
                _render_dashboard_html(empty_obs, demo_session["task_id"], demo_session["difficulty"], str(demo_session["seed"]), "0.0000", "0.0000", False, {"components": {}}, demo_session["history"]),
                schema_rows,
                kpi_rows,
                _empty_reward_rows(),
                "No episode is active yet.",
                "Initialize the environment first to execute steps.",
                {"error": "System not initialized. Reset the environment first."},
            )

        try:
            params = json.loads(params_json) if params_json.strip() else {}
        except json.JSONDecodeError:
            params = {}

        cols = [c.strip() for c in target_cols.split(",") if c.strip()] if target_cols else []
        action = Action(action_type=action_type, target_columns=cols, parameters=params, reasoning=reasoning or "System execution")
        obs, reward, done, info = demo_env.step(action)
        demo_session["obs"] = obs
        demo_session["history"].append({"action": action_type, "reward": reward.value})

        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
            "task_enterprise_orchestration": EnterpriseOrchestrationGrader,
        }
        grade = graders[demo_session["task_id"]].grade(demo_env.current_episode)
        schema_rows, kpi_rows = _observation_tables(obs)
        reward_rows = _build_reward_rows(obs, reward.value, grade, info)
        raw_payload = _build_raw_payload(obs, reward.value, grade, done, info, {
            "action_type": action_type,
            "target_columns": cols,
            "parameters": params,
            "reasoning": reasoning or "System execution",
        })

        history_line = " → ".join(f"{item['action']}({item['reward']:.3f})" for item in demo_session["history"])
        if not history_line:
            history_line = "No execution history recorded yet."

        guidance_text = "\n".join(f"- {step}" for step in _suggest_next_steps(obs, demo_session["history"]))

        return (
            _render_dashboard_html(obs, demo_session["task_id"], demo_session["difficulty"], str(demo_session["seed"]), f"{reward.value:.4f}", f"{grade:.4f}", done, info, demo_session["history"]),
            schema_rows,
            kpi_rows,
            reward_rows,
            history_line,
            guidance_text,
            raw_payload,
        )

    custom_css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', sans-serif;
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.18), transparent 36%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .dark .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 36%),
            radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 30%),
            linear-gradient(180deg, #020617 0%, #0f172a 100%);
    }
    .hero, .panel {
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(16px);
    }
    .hero {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-end;
        padding: 24px;
        border-radius: 24px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.9));
        color: white;
    }
    .hero h2 {
        margin: 6px 0 10px;
        font-size: 2.1rem;
        line-height: 1.05;
    }
    .hero p {
        margin: 0;
        color: rgba(226, 232, 240, 0.9);
        max-width: 62ch;
    }
    .eyebrow {
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: rgba(191, 219, 254, 0.95);
    }
    .hero-badges {
        display: grid;
        gap: 10px;
        min-width: 220px;
    }
    .badge {
        border-radius: 999px;
        padding: 10px 14px;
        background: rgba(255, 255, 255, 0.08);
        color: rgba(241, 245, 249, 0.96);
        font-weight: 600;
        text-align: center;
    }
    .badge-primary {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.28), rgba(59, 130, 246, 0.32));
    }
    .panel {
        border-radius: 22px;
        padding: 20px;
        margin-bottom: 16px;
        background: rgba(255, 255, 255, 0.76);
    }
    .dark .panel {
        background: rgba(15, 23, 42, 0.72);
    }
    .panel-main { overflow: hidden; }
    .panel-soft {
        margin-bottom: 14px;
        background: rgba(255, 255, 255, 0.55);
    }
    .dark .panel-soft {
        background: rgba(15, 23, 42, 0.54);
    }
    .panel-title {
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
        color: #0f172a;
    }
    .dark .panel-title { color: #e2e8f0; }
    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 14px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(148, 163, 184, 0.12);
        color: #0f172a;
        font-size: 0.86rem;
        font-weight: 600;
    }
    .dark .status-pill { color: #e2e8f0; }
    .status-live { background: rgba(14, 165, 233, 0.16); }
    .status-done { background: rgba(16, 185, 129, 0.18); }
    .stat-grid, .two-col, .signal-grid {
        display: grid;
        gap: 12px;
    }
    .stat-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 16px; }
    .two-col { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 16px; }
    .signal-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 12px; }
    .stat-card, .signal-card, .drift-box {
        border-radius: 18px;
        padding: 16px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(255, 255, 255, 0.78);
    }
    .dark .stat-card, .dark .signal-card, .dark .drift-box {
        background: rgba(15, 23, 42, 0.6);
    }
    .stat-label, .signal-label {
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .dark .stat-label, .dark .signal-label { color: #94a3b8; }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .dark .stat-value { color: #f8fafc; }
    .stat-note {
        margin-top: 6px;
        color: #64748b;
        font-size: 0.82rem;
    }
    .dark .stat-note { color: #94a3b8; }
    .narrative {
        padding: 16px;
        border-radius: 16px;
        background: rgba(14, 165, 233, 0.08);
        color: #0f172a;
        line-height: 1.65;
        margin-bottom: 16px;
        white-space: pre-wrap;
    }
    .dark .narrative { color: #e2e8f0; background: rgba(14, 165, 233, 0.12); }
    .signal-body {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        line-height: 1.6;
        color: #0f172a;
    }
    .dark .signal-body { color: #e2e8f0; }
    .chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 6px 10px;
        background: rgba(59, 130, 246, 0.12);
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .dark .chip { color: #f8fafc; background: rgba(59, 130, 246, 0.16); }
    .chip-muted { background: rgba(148, 163, 184, 0.16); }
    .bullet-list {
        margin: 0;
        padding-left: 20px;
        color: #0f172a;
    }
    .dark .bullet-list { color: #e2e8f0; }
    .bullet-list li { margin-bottom: 8px; }
    .table-wrap {
        overflow-x: auto;
    }
    .dash-table {
        width: 100%;
        border-collapse: collapse;
        color: #0f172a;
        font-size: 0.92rem;
    }
    .dark .dash-table { color: #e2e8f0; }
    .dash-table th, .dash-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        text-align: left;
        vertical-align: top;
    }
    .dash-table th {
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
    }
    .dark .dash-table th { color: #94a3b8; }
    .mini-card {
        border-radius: 14px;
        padding: 12px 14px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .dark .mini-card { background: rgba(15, 23, 42, 0.72); }
    .mini-card span {
        display: block;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .mini-card strong { color: #0f172a; }
    .dark .mini-card strong { color: #f8fafc; }
    .gr-button-primary {
        background: linear-gradient(135deg, #0f172a, #1d4ed8) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: 0 !important;
        box-shadow: 0 14px 30px rgba(29, 78, 216, 0.28);
    }
    .gr-button-secondary {
        background: linear-gradient(135deg, #475569, #334155) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: 0 !important;
    }
    """

    with gr.Blocks(title="Enterprise Orchestration Console") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        gr.Markdown("# Enterprise Orchestration Console")
        gr.Markdown("*Refined live demo for world modeling, multi-agent negotiation, and drift-aware execution.*")

        with gr.Tabs():
            with gr.TabItem("Live Console"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("### Episode Controls")
                        task_dd = gr.Dropdown(
                            choices=["task_enterprise_orchestration", "task_missing_values", "task_duplicate_handling", "task_complex_validation"],
                            value="task_enterprise_orchestration",
                            label="Task Identifier",
                        )
                        diff_dd = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Simulation Difficulty")
                        seed_tb = gr.Textbox(value="42", label="Random Seed")
                        reset_btn = gr.Button("Initialize System", variant="primary")

                        gr.Markdown("### Action Composer")
                        action_dd = gr.Dropdown(
                            choices=["analyze", "impute", "deduplicate", "validate", "report_findings", "delegate", "resolve_alert", "reconcile_apps", "oversight_review", "inspect_actor", "audit_records", "request_policy_clarification"],
                            value="analyze",
                            label="Action Type",
                        )
                        cols_tb = gr.Textbox(label="Target Columns", placeholder="account_id, invoice_status")
                        params_tb = gr.Textbox(label="Action Parameters (JSON)", value="{}", lines=3)
                        reason_tb = gr.Textbox(label="Strategic Reasoning", placeholder="Explain why this step is appropriate.", lines=4)
                        step_btn = gr.Button("Execute Step", variant="secondary")

                        gr.Markdown("### Episode Guidance")
                        guidance_box = gr.Markdown("Initialize an episode to see tactical guidance.")

                    with gr.Column(scale=2, min_width=560):
                        summary_html = gr.HTML(value="<div class='panel panel-main'><div class='panel-title'>Awaiting initialization</div><div class='narrative'>Press Initialize System to load the live environment dashboard.</div></div>")
                        with gr.Row():
                            schema_df = gr.Dataframe(headers=["Column", "Type", "Missing"], interactive=False, label="Schema Snapshot")
                            kpi_df = gr.Dataframe(headers=["KPI", "Value"], interactive=False, label="KPI Snapshot")
                        reward_df = gr.Dataframe(headers=["Signal", "Value"], interactive=False, label="Reward & Budget")
                        history_md = gr.Markdown("No execution history recorded yet.")
                        raw_json = gr.JSON(label="Raw Observation Payload")

                reset_btn.click(reset_env, inputs=[task_dd, diff_dd, seed_tb], outputs=[summary_html, schema_df, kpi_df, reward_df, history_md, guidance_box, raw_json])
                step_btn.click(step_env, inputs=[action_dd, cols_tb, params_tb, reason_tb], outputs=[summary_html, schema_df, kpi_df, reward_df, history_md, guidance_box, raw_json])

            with gr.TabItem("Training Evidence"):
                gr.Markdown("### Reinforcement Learning Evidence")
                gr.Markdown("These charts document the training run that produced the policy used in the demo.")

                import os
                base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                def _find(name):
                    local = os.path.join("artifacts", name)
                    if os.path.exists(local):
                        return local
                    full = os.path.join(base, "artifacts", name)
                    if os.path.exists(full):
                        return full
                    return None

                rp = _find("reward_progression_chart.png")
                tb = _find("task_breakdown_chart.png")
                ab = _find("ablation_chart.png")
                flow = _find("world_model_flow.svg")
                traj = _find("failure_success_trajectory.svg")

                with gr.Row():
                    if rp:
                        gr.Image(value=rp, type="filepath", label="Reward Progression")
                    if tb:
                        gr.Image(value=tb, type="filepath", label="Per-Task Score Improvement")
                with gr.Row():
                    if ab:
                        gr.Image(value=ab, type="filepath", label="Ablation: Actor Actions Impact")
                    if traj:
                        gr.Image(value=traj, type="filepath", label="Failure vs Success Trajectory")
                if flow:
                    gr.Image(value=flow, type="filepath", label="World Model Flow")
                if not any([rp, tb, ab, flow, traj]):
                    gr.Markdown("*(Training charts not found. Run `python training/generate_charts.py` to generate them.)*")

            with gr.TabItem("Method & Metrics"):
                gr.Markdown("""### What this environment teaches

An LLM agent is dropped into a simulated enterprise with three connected systems and five autonomous actors whose incentives conflict. The task is to reason over messy state, react to drift, and sequence actions professionally.

### Core behaviors

- Negotiate with actors when their goals conflict.
- Detect deception before following untrusted recommendations.
- Adapt to schema drift and revised validation rules.
- Manage budget pressure while preserving data quality.
- Reason before acting so the reward model can distinguish process quality from brute force.

### Anti-gaming safeguards

- Loop penalties prevent action spam.
- Reasoning quality checks penalize empty rationales.
- Report rewards require real data improvement.
- Policy clarification only pays once per version.
- Drift penalties escalate until the system is handled correctly.

### Key metrics

| Metric | Value |
|--------|-------|
| Baseline score | 0.488 |
| Trained score | **0.701 (+43.8%)** |
| Ablation delta (actor actions) | **+0.384** |
| Held-out hard drift score | **0.831** |
""")

    return demo


# Mount Gradio
try:
    import gradio as gr
    demo = _build_gradio_demo()
    if demo is not None:
        app = gr.mount_gradio_app(app, demo, path="/demo")
except ImportError:
    pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
