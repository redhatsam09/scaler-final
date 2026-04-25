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
        import matplotlib.pyplot as plt
        from pathlib import Path
        import time
    except ImportError:
        return None

    demo_env = DataCleaningEnv(seed=42)
    demo_session = {"obs": None, "history": [], "task_id": "task_enterprise_orchestration"}

    base_dir = Path(__file__).resolve().parents[1]

    def _artifact_path(name: str) -> str | None:
        local = base_dir / "artifacts" / name
        if local.exists():
            return str(local)
        return None

    def _history_figure():
        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#0f172a")
        ax.grid(True, alpha=0.15, color="#94a3b8")
        
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

        if not demo_session["history"]:
            ax.text(0.5, 0.5, "No steps executed yet", ha="center", va="center", transform=ax.transAxes, color="#cbd5e1")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Reward Trend", fontsize=10, color="#f8fafc")
            fig.tight_layout()
            return fig

        xs = list(range(1, len(demo_session["history"]) + 1))
        ys = [float(item["reward"]) for item in demo_session["history"]]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color="#38bdf8")
        ax.set_xlabel("Step", color="#cbd5e1", fontsize=9)
        ax.set_ylabel("Reward", color="#cbd5e1", fontsize=9)
        ax.set_title("Reward Trend", fontsize=10, color="#f8fafc")
        fig.tight_layout()
        return fig

    def _format_kpi_rows(obs):
        if not obs.kpi_snapshot:
            return [["No KPI data", "-"]]
        return [[k, f"{v:.4f}"] for k, v in obs.kpi_snapshot.items()]

    def _format_reward_rows(info: Dict[str, Any], reward_value: float, grade_value: float):
        rows = [["step_reward", f"{reward_value:.4f}"], ["cumulative_grade", f"{grade_value:.4f}"]]
        components = info.get("components", {}) if isinstance(info, dict) else {}
        for key, value in components.items():
            rows.append([key, f"{float(value):.4f}"])
        return rows

    def reset_env(task_id, difficulty, seed):
        seed_val = int(seed) if seed else 42
        obs = demo_env.reset(task_id=task_id, seed=seed_val, difficulty=difficulty)
        demo_session["obs"] = obs
        demo_session["history"] = []
        demo_session["task_id"] = task_id

        state_text = obs.natural_language_observation
        kpi_rows = _format_kpi_rows(obs)
        reward_rows = [["step_reward", "0.0000"], ["cumulative_grade", "0.0000"]]
        urgency = "".join(f"<li style='color: #ef4444'>{s}</li>" for s in obs.urgency_signals) if obs.urgency_signals else "<li>None</li>"
        actors = "".join(f"<li>{m}</li>" for m in obs.actor_messages) if obs.actor_messages else "<li>None</li>"

        output = f"""<div style='padding: 16px; border-radius: 10px; background: var(--background-fill-secondary); color: var(--body-text-color); border: 1px solid var(--border-color-primary); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
<h3 style='margin-top: 0; color: var(--color-accent); font-weight: 600;'>Environment Reset Successfully</h3>
<p style='margin-bottom: 8px;'><b>Task:</b> {task_id} &nbsp;|&nbsp; <b>Difficulty:</b> {difficulty} &nbsp;|&nbsp; <b>Seed:</b> {seed_val}</p>
<p style='margin-bottom: 12px;'><b>Dataset:</b> {obs.dataset_shape[0]} rows × {obs.dataset_shape[1]} cols</p>
<hr style='border-color: var(--border-color-primary); margin: 12px 0;'>
<p style='line-height: 1.6;'><b>Observation:</b> {state_text}</p>
<p style='margin-top: 12px;'><b>Urgency Signals:</b><ul style='margin-top: 4px;'>{urgency}</ul></p>
<p><b>Actor Messages:</b><ul style='margin-top: 4px;'>{actors}</ul></p>
<p style='margin-top: 12px; font-size: 0.9em; color: var(--body-text-color-subdued);'><b>Available Actions:</b> {', '.join(obs.available_actions)}</p>
</div>"""
        return output, "", kpi_rows, reward_rows, _history_figure()

    def step_env(action_type, target_cols, params_json, reasoning):
        if demo_session["obs"] is None:
            return "<div style='color: #ef4444'>Reset the environment first.</div>", "", [["No KPI data", "-"]], [["step_reward", "0.0000"]], _history_figure()

        try:
            params = json.loads(params_json) if params_json.strip() else {}
        except json.JSONDecodeError:
            params = {}

        cols = [c.strip() for c in target_cols.split(",") if c.strip()] if target_cols else []
        action = Action(action_type=action_type, target_columns=cols, parameters=params, reasoning=reasoning or "Manual step")
        obs, reward, done, info = demo_env.step(action)
        demo_session["obs"] = obs
        demo_session["history"].append({"action": action_type, "reward": reward.value})

        kpi_rows = _format_kpi_rows(obs)
        urgency = "".join(f"<li style='color: #ef4444'>{s}</li>" for s in obs.urgency_signals) if obs.urgency_signals else "<li>None</li>"
        actors = "".join(f"<li>{m}</li>" for m in obs.actor_messages[-3:]) if obs.actor_messages else "<li>None</li>"

        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
            "task_enterprise_orchestration": EnterpriseOrchestrationGrader,
        }
        grade = graders[demo_session["task_id"]].grade(demo_env.current_episode)
        reward_rows = _format_reward_rows(info, reward.value, grade)

        history_text = " → ".join(f"{h['action']}({h['reward']:.2f})" for h in demo_session["history"][-6:])

        status_color = "#22c55e" if done else "var(--color-accent)"
        status_text = "EPISODE COMPLETE" if done else f"STEP {obs.step_count}"
        step_num = len(demo_session["history"])
        reward_color = "#22c55e" if reward.value > 0.05 else ("#ef4444" if reward.value < -0.01 else "#f59e0b")

        output = f"""<div style='padding: 18px; border-radius: 10px; background: var(--background-fill-secondary); color: var(--body-text-color); border: 1px solid var(--border-color-primary); border-left: 5px solid {status_color}; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
<h3 style='margin-top: 0; color: {status_color}; letter-spacing: 1px; font-size: 1.1em; font-weight: 700;'>{status_text}</h3>
<div style='display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;'>
  <span style='background: var(--background-fill-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.9em; border: 1px solid var(--border-color-primary);'>ACTION: <b>{action_type}</b></span>
  <span style='background: var(--background-fill-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.9em; color: {reward_color}; border: 1px solid var(--border-color-primary);'>REWARD: <b>{reward.value:.4f}</b></span>
  <span style='background: var(--background-fill-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.9em; border: 1px solid var(--border-color-primary);'>GRADE: <b>{grade:.4f}</b></span>
  <span style='background: var(--background-fill-primary); padding: 4px 10px; border-radius: 6px; font-size: 0.9em; color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary);'>STEP {step_num}</span>
</div>
<hr style='border-color: var(--border-color-primary); margin: 12px 0;'>
<p style='line-height: 1.6;'><b>Observation:</b> {obs.natural_language_observation}</p>
<p style='margin-top: 12px;'><b>Urgency Signals:</b><ul style='margin: 4px 0;'>{urgency}</ul></p>
<p><b>Actor Messages:</b><ul style='margin: 4px 0;'>{actors}</ul></p>
<p style='margin-top: 14px; color: var(--body-text-color-subdued); font-size: 0.9em;'><b>Action History:</b> {history_text}</p>
</div>"""
        return output, "", kpi_rows, reward_rows, _history_figure()

    def auto_play(task_id):
        """Auto-play expert policy with narrated explanations and slow pacing."""
        out, err, kpi, rew, fig = reset_env(task_id, "medium", "42")
        yield out, err, kpi, rew, fig
        
        # Each tuple: (action, cols, params, reasoning, narration_for_judge)
        sequences = {
            "task_enterprise_orchestration": [
                ("analyze", "", "{}", "Profile data quality before any changes",
                 "[1/7 ANALYZE] Always analyze first. The agent inspects data quality metrics to plan its strategy."),
                ("inspect_actor", "account_id", '{"actor": "finance_bot"}', "Check finance bot trust before delegating",
                 "[2/7 INSPECT] The agent checks if finance_bot is trustworthy BEFORE delegating work. This is core world modeling — reasoning about hidden state."),
                ("delegate", "account_id", '{"actor": "finance_bot", "objective": "invoice cleanup"}', "Delegate invoice work after confirming trust",
                 "[3/7 DELEGATE] Trust confirmed. The agent assigns invoice cleanup to finance_bot. Note: delegation is stochastic — the actor might push back."),
                ("oversight_review", "account_id", '{"actor": "analytics_assistant", "explain": true}', "Detect deceptive recommendations from analytics assistant",
                 "[4/7 OVERSIGHT] The analytics_assistant may have recommended a compliance-violating shortcut. The agent runs an oversight review to detect deception."),
                ("reconcile_apps", "account_id", '{"join_key": "account_id"}', "Fix cross-app data conflicts between CRM and Billing",
                 "[5/7 RECONCILE] CRM, Billing, and Support data can disagree on the same account. The agent patches cross-system conflicts."),
                ("validate", "compliance_tier", '{"compliance_tier_type": "categorical_nonempty"}', "Validate after schema drift changed compliance rules",
                 "[6/7 VALIDATE] Schema drift may have introduced new compliance rules. The agent validates against the latest policy version."),
                ("report_findings", "account_id", '{"include_summary": true, "include_quality_score": true, "include_recommendations": true}', "Final quality report with all improvements",
                 "[7/7 REPORT] Final report. Reward only fires if actual data quality improved — this prevents gaming via empty reports.")
            ],
            "task_missing_values": [
                ("analyze", "", "{}", "Analyze dataset for missing values",
                 "[1/5 ANALYZE] Profile which columns have missing values and how severe the gaps are."),
                ("impute", "email,phone", '{"method": "forward_fill"}', "Fill gaps in text columns using forward fill",
                 "[2/5 IMPUTE] Forward-fill propagates the last known value. The reward tracks the reduction in missing values."),
                ("impute", "lead_score", '{"method": "mean"}', "Fill numeric gaps using column mean",
                 "[3/5 IMPUTE] Mean imputation for numeric fields preserves the column distribution."),
                ("validate", "email", "{}", "Validate cleaned data",
                 "[4/5 VALIDATE] Check data types and constraints after imputation."),
                ("report_findings", "email", '{"include_summary": true, "include_quality_score": true}', "Report findings",
                 "[5/5 REPORT] Final summary. Quality score must exceed the baseline or the report earns reduced reward.")
            ],
            "task_duplicate_handling": [
                ("analyze", "", "{}", "Profile duplicates in dataset",
                 "[1/4 ANALYZE] Scan for duplicate records in the invoice dataset."),
                ("deduplicate", "invoice_id", '{"subset": ["invoice_id"], "keep": "first"}', "Remove duplicates by invoice_id",
                 "[2/4 DEDUPLICATE] Remove duplicate invoice records. Note: over-deletion is penalized."),
                ("validate", "invoice_id,amount", "{}", "Validate deduplication results",
                 "[3/4 VALIDATE] Confirm no data corruption occurred after deduplication."),
                ("report_findings", "invoice_id", '{"include_summary": true, "include_quality_score": true}', "Report deduplication results",
                 "[4/4 REPORT] Document the deduplication outcome with quality metrics.")
            ],
            "task_complex_validation": [
                ("analyze", "", "{}", "Analyze complex validation requirements",
                 "[1/6 ANALYZE] Understand the multi-constraint validation landscape."),
                ("impute", "email,phone", '{"method": "forward_fill"}', "Fix missing values before validation",
                 "[2/6 IMPUTE] Fill gaps so validation rules can be properly checked."),
                ("deduplicate", "account_id", '{"keep": "first"}', "Remove duplicate accounts",
                 "[3/6 DEDUPLICATE] Clean duplicates before cross-app reconciliation."),
                ("reconcile_apps", "account_id", '{"join_key": "account_id"}', "Reconcile cross-app data",
                 "[4/6 RECONCILE] Align CRM, Billing, Support data for this account."),
                ("validate", "csat_score", '{"csat_score_type": "numeric", "csat_score_min": 1, "csat_score_max": 5}', "Validate with constraints",
                 "[5/6 VALIDATE] Check numeric ranges, categorical constraints, and cross-field rules."),
                ("report_findings", "account_id", '{"include_summary": true, "include_quality_score": true}', "Report validation results",
                 "[6/6 REPORT] Summarize all validation findings and quality improvement.")
            ]
        }
        
        seq = sequences.get(task_id, sequences["task_missing_values"])
        for action, cols, params, reasoning, narration in seq:
            time.sleep(3.0)
            out, err, kpi, rew, fig = step_env(action, cols, params, reasoning)
            narration_html = f"""<div style='padding: 12px 16px; margin-bottom: 12px; border-radius: 8px; background: var(--background-fill-secondary); border: 2px solid var(--color-accent); color: var(--body-text-color); font-size: 0.95em; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
<strong style='color: var(--color-accent); display: flex; align-items: center; gap: 8px;'><svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><circle cx='12' cy='12' r='10'/><path d='M12 16v-4'/><path d='M12 8h.01'/></svg> AGENT REASONING:</strong> 
<span style='display: inline-block; margin-top: 6px;'>{narration}</span>
</div>"""
            out = narration_html + out
            yield out, err, kpi, rew, fig

    def preset_action(action_type):
        presets = {
            "analyze": ("", "{}", "Initial profile of the data quality"),
            "impute": ("email,phone", '{"method": "forward_fill"}', "Fill missing values to improve quality index"),
            "deduplicate": ("invoice_id", '{"keep": "first"}', "Remove duplicate records to prevent overbilling"),
            "validate": ("compliance_tier", '{"compliance_tier_type": "categorical_nonempty"}', "Ensure compliance rules are met"),
            "delegate": ("", '{"actor": "finance_bot", "objective": "invoice cleanup"}', "Assign domain work to expert actor"),
            "inspect_actor": ("", '{"actor": "finance_bot"}', "Reveal actor trust and hidden objectives"),
            "oversight_review": ("", '{"actor": "analytics_assistant", "explain": true}', "Check for deceptive shortcut recommendations"),
            "reconcile_apps": ("account_id", '{"join_key": "account_id"}', "Patch cross-system conflicts"),
            "report_findings": ("", '{"include_summary": true, "include_quality_score": true}', "Final step to summarize improvements")
        }
        return presets.get(action_type, ("", "{}", "Execute action"))

    premium_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Enterprise Orchestration Lab", theme=premium_theme) as demo:
        gr.HTML("""
        <div style="text-align: center; max-width: 860px; margin: 0 auto; padding: 28px 0 16px;">
            <div style="display: inline-flex; align-items: center; gap: 14px; margin-bottom: 12px;">
                <div style="width: 44px; height: 44px; background: linear-gradient(135deg, #2563eb, #7c3aed); border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2'><path d='M12 2L2 7l10 5 10-5-10-5z'/><path d='M2 17l10 5 10-5'/><path d='M2 12l10 5 10-5'/></svg>
                </div>
                <h1 style="color: var(--body-text-color); font-size: 2.2em; margin: 0; font-weight: 700; letter-spacing: -0.5px;">Enterprise Orchestration Lab</h1>
            </div>
            <p style="color: var(--body-text-color-subdued); font-size: 1.0em; line-height: 1.7; max-width: 700px; margin: 0 auto;">
                A multi-system RL environment for <strong style="color:var(--color-accent)">World Modeling</strong> (Theme 3.1).
                Agents manage CRM, Billing, and Support while navigating schema drift, actor conflicts, deceptive oversight, and economic budgets.
            </p>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 14px; flex-wrap: wrap;">
                <span style="background: var(--background-fill-secondary); color: var(--color-accent); padding: 4px 12px; border-radius: 20px; font-size: 0.8em; border: 1px solid var(--border-color-primary);">REINFORCE Policy Gradient</span>
                <span style="background: var(--background-fill-secondary); color: var(--color-accent); padding: 4px 12px; border-radius: 20px; font-size: 0.8em; border: 1px solid var(--border-color-primary);">Qwen 2.5-1.5B + LoRA</span>
                <span style="background: var(--background-fill-secondary); color: var(--color-accent); padding: 4px 12px; border-radius: 20px; font-size: 0.8em; border: 1px solid var(--border-color-primary);">4-bit NF4 Quantization</span>
                <span style="background: var(--background-fill-secondary); color: var(--color-accent); padding: 4px 12px; border-radius: 20px; font-size: 0.8em; border: 1px solid var(--border-color-primary);">Environment-Grounded Rewards</span>
            </div>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Simulation Console"):
                gr.HTML("""
                <div style='background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;'>
                    <strong style='color: var(--body-text-color);'>Quick Start:</strong>
                    <span style='color: var(--body-text-color-subdued); font-size: 0.9em;'>1️⃣ Click <b>Reset Environment</b></span> <span style='color: var(--color-accent);'>→</span>
                    <span style='color: var(--body-text-color-subdued); font-size: 0.9em;'>2️⃣ Check <b>Environment Output</b> state</span> <span style='color: var(--color-accent);'>→</span>
                    <span style='color: var(--body-text-color-subdued); font-size: 0.9em;'>3️⃣ Pick <b>Action Type</b> (Params auto-fill! 🎯)</span> <span style='color: var(--color-accent);'>→</span>
                    <span style='color: var(--body-text-color-subdued); font-size: 0.9em;'>4️⃣ Write <b>Reasoning</b> & Execute Step!</span>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=1, variant="panel"):
                        gr.HTML("<h3 style='color:var(--body-text-color); margin:0 0 8px;'>Session Configuration</h3>")
                        task_dd = gr.Dropdown(
                            choices=["task_enterprise_orchestration", "task_missing_values",
                                     "task_duplicate_handling", "task_complex_validation"],
                            value="task_enterprise_orchestration", label="Task Scenario",
                            info="Enterprise Orchestration is the flagship task with schema drift, actor conflicts, and deceptive oversight"
                        )
                        with gr.Row():
                            diff_dd = gr.Dropdown(choices=["easy", "medium", "hard"], value="hard", label="Difficulty")
                            seed_tb = gr.Textbox(value="42", label="Random Seed")
                        
                        with gr.Row():
                            reset_btn = gr.Button("Reset Environment", variant="primary", size="lg")
                            autoplay_btn = gr.Button("Watch Expert Policy", variant="secondary", size="lg")
                        gr.HTML("<p style='color:var(--body-text-color-subdued); font-size:0.82em; margin:4px 0 0;'>Expert Policy runs a 7-step narrated demo showing optimal agent reasoning.</p>")

                        gr.HTML("<hr style='border-color:var(--border-color-primary); margin:14px 0;'>")
                        gr.HTML("<h3 style='color:var(--body-text-color); margin:0 0 8px;'>Manual Action Execution</h3>")
                        action_dd = gr.Dropdown(
                            choices=["analyze", "impute", "deduplicate", "validate", "report_findings",
                                     "delegate", "resolve_alert", "reconcile_apps", "oversight_review",
                                     "inspect_actor", "audit_records", "request_policy_clarification"],
                            value="analyze", label="Action Type", 
                            info="Select an action and parameters auto-fill with recommended defaults"
                        )
                        cols_tb = gr.Textbox(label="Target Columns", placeholder="e.g. account_id, invoice_status",
                                            info="Comma-separated column names from the dataset")
                        params_tb = gr.Code(label="Parameters (JSON)", value="{}", language="json", lines=2)
                        reason_tb = gr.Textbox(label="Reasoning", placeholder="e.g. Analyze data quality before making changes",
                                             info="Explain your decision (15+ chars). Short reasoning is penalized.")
                        step_btn = gr.Button("Execute Step", variant="primary", size="lg")
                        
                        action_dd.change(preset_action, inputs=[action_dd], outputs=[cols_tb, params_tb, reason_tb])

                    with gr.Column(scale=2):
                        gr.HTML("<h3 style='color:var(--body-text-color); margin:0 0 8px;'>Environment Output</h3>")
                        output_html = gr.HTML("""<div style='padding: 30px; text-align: center; color: var(--body-text-color-subdued); border: 1px dashed var(--border-color-primary); border-radius: 10px; background: var(--background-fill-secondary);'>
                            <svg width='40' height='40' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' style='margin-bottom:10px'><circle cx='12' cy='12' r='10'/><path d='M12 8v4l3 3'/></svg>
                            <p style='margin:0; font-size:1.05em;'>Click <b style='color:var(--color-accent)'>Reset Environment</b> to start a session or <b>Watch Expert Policy</b> for a narrated demo.</p>
                        </div>""")
                        error_md = gr.Markdown("")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                kpi_df = gr.Dataframe(
                                    headers=["KPI Metric", "Value"],
                                    datatype=["str", "str"],
                                    label="Live KPI Snapshot",
                                    interactive=False,
                                )
                            with gr.Column(scale=1):
                                reward_df = gr.Dataframe(
                                    headers=["Reward Component", "Value"],
                                    datatype=["str", "str"],
                                    label="Reward Breakdown",
                                    interactive=False,
                                )
                        history_plot = gr.Plot(label="Step Reward Trend")

                reset_btn.click(
                    reset_env,
                    inputs=[task_dd, diff_dd, seed_tb],
                    outputs=[output_html, error_md, kpi_df, reward_df, history_plot],
                )
                step_btn.click(
                    step_env,
                    inputs=[action_dd, cols_tb, params_tb, reason_tb],
                    outputs=[output_html, error_md, kpi_df, reward_df, history_plot],
                )
                autoplay_btn.click(
                    auto_play,
                    inputs=[task_dd],
                    outputs=[output_html, error_md, kpi_df, reward_df, history_plot],
                )

            with gr.Tab("Training Evidence"):
                gr.HTML("""<div style='padding:16px; background:var(--background-fill-secondary); border-radius:10px; margin-bottom:12px; border:1px solid var(--border-color-primary);'>
                    <h3 style='color:var(--body-text-color); margin:0 0 8px;'>Training Pipeline Evidence</h3>
                    <p style='color:var(--body-text-color-subdued); margin:0; line-height:1.6;'>
                        Model trained with <b style='color:var(--color-accent)'>REINFORCE policy gradient</b> on Qwen 2.5-1.5B-Instruct (4-bit NF4 + LoRA).
                        Each step generates a JSON action, runs it through <code style='background:var(--background-fill-primary); padding:2px 6px; border-radius:4px; border:1px solid var(--border-color-primary);'>env.step()</code>,
                        and uses the environment reward as the training signal. No proxy rewards.
                    </p>
                </div>""")
                
                with gr.Row():
                    with gr.Column():
                        tc = _artifact_path("training_curves.svg")
                        if tc: gr.Image(value=tc, type="filepath", label="Training Loss and Reward Curves", show_label=True)
                        else: gr.HTML("<p style='color:var(--body-text-color-subdued); text-align:center;'>Run the Colab notebook to generate training curves.</p>")
                    with gr.Column():
                        ev = _artifact_path("eval_results.svg")
                        if ev: gr.Image(value=ev, type="filepath", label="Evaluation Results (12 Episodes)", show_label=True)
                        else: gr.HTML("<p style='color:var(--body-text-color-subdued); text-align:center;'>Run the Colab notebook to generate evaluation results.</p>")
                
                with gr.Row():
                    with gr.Column():
                        rp = _artifact_path("reward_progression.svg")
                        if rp: gr.Image(value=rp, type="filepath", label="Reward Progression (Baseline vs Trained)", show_label=True)
                    with gr.Column():
                        flow = _artifact_path("world_model_flow.svg")
                        if flow: gr.Image(value=flow, type="filepath", label="World Model Architecture Flow", show_label=True)

            with gr.Tab("Methodology"):
                gr.HTML("""<div style='max-width:800px; margin:0 auto; padding:20px;'>
                    <h2 style='color:var(--body-text-color); margin-bottom:16px;'>Training Methodology</h2>
                    
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px;'>
                        <div style='background:var(--background-fill-secondary); padding:16px; border-radius:10px; border-left:3px solid var(--color-accent);'>
                            <h4 style='color:var(--color-accent); margin:0 0 8px;'>Model</h4>
                            <p style='color:var(--body-text-color-subdued); margin:0; font-size:0.95em;'>Qwen 2.5-1.5B-Instruct with LoRA (r=16) adapters on all attention and MLP projections. 4-bit NF4 quantization via bitsandbytes.</p>
                        </div>
                        <div style='background:var(--background-fill-secondary); padding:16px; border-radius:10px; border-left:3px solid #22c55e;'>
                            <h4 style='color:#22c55e; margin:0 0 8px;'>Training Algorithm</h4>
                            <p style='color:var(--body-text-color-subdued); margin:0; font-size:0.95em;'>REINFORCE with running baseline. Loss = -(reward - baseline) * mean_log_prob. Gradient clipping at 1.0. AdamW with lr=2e-5.</p>
                        </div>
                        <div style='background:var(--background-fill-secondary); padding:16px; border-radius:10px; border-left:3px solid #f59e0b;'>
                            <h4 style='color:#f59e0b; margin:0 0 8px;'>Reward Function</h4>
                            <p style='color:var(--body-text-color-subdued); margin:0; font-size:0.95em;'>Multi-level scoring: JSON format (-1.0 to -0.3), key completeness, fuzzy action_type matching (30+ synonyms mapped), env.step() execution reward + grader score.</p>
                        </div>
                        <div style='background:var(--background-fill-secondary); padding:16px; border-radius:10px; border-left:3px solid #a855f7;'>
                            <h4 style='color:#a855f7; margin:0 0 8px;'>Environment Dynamics</h4>
                            <p style='color:var(--body-text-color-subdued); margin:0; font-size:0.95em;'>Schema drift (policy v1-v3), 5 actors with hidden trust scores, deceptive recommendations, cross-app data conflicts, action costs with budget limits.</p>
                        </div>
                    </div>
                    
                    <h3 style='color:var(--body-text-color); margin-bottom:12px;'>Training Pipeline</h3>
                    <div style='background:var(--background-fill-secondary); padding:16px; border-radius:10px; border:1px solid var(--border-color-primary);'>
                        <p style='color:var(--body-text-color-subdued); line-height:1.8; margin:0; font-family:monospace; font-size:0.9em;'>
                            <span style='color:var(--color-accent);'>1.</span> Sample environment state as prompt<br>
                            <span style='color:var(--color-accent);'>2.</span> Model generates JSON action (temp=1.0, top_p=0.95)<br>
                            <span style='color:var(--color-accent);'>3.</span> Parse JSON with fuzzy action_type matching<br>
                            <span style='color:var(--color-accent);'>4.</span> Execute action via env.step() for verifiable reward<br>
                            <span style='color:var(--color-accent);'>5.</span> Compute log-probs of generated tokens under current policy<br>
                            <span style='color:var(--color-accent);'>6.</span> REINFORCE update: loss = -(reward - baseline) * mean_log_prob<br>
                            <span style='color:var(--color-accent);'>7.</span> Clip gradients and update LoRA weights<br>
                        </p>
                    </div>
                    
                    <h3 style='color:var(--body-text-color); margin:16px 0 12px;'>World Modeling Capabilities Tested</h3>
                    <table style='width:100%; border-collapse:collapse; color:var(--body-text-color-subdued); font-size:0.9em;'>
                        <tr style='border-bottom:1px solid var(--border-color-primary);'><td style='padding:8px; color:var(--color-accent);'>Partial Observability</td><td style='padding:8px;'>Actor trust scores are hidden; agents must inspect before delegating</td></tr>
                        <tr style='border-bottom:1px solid var(--border-color-primary);'><td style='padding:8px; color:#22c55e;'>Schema Drift</td><td style='padding:8px;'>Policy versions change compliance rules mid-episode</td></tr>
                        <tr style='border-bottom:1px solid var(--border-color-primary);'><td style='padding:8px; color:#f59e0b;'>Deceptive Actors</td><td style='padding:8px;'>Analytics assistant may recommend compliance-violating shortcuts</td></tr>
                        <tr style='border-bottom:1px solid var(--border-color-primary);'><td style='padding:8px; color:#a855f7;'>Multi-Stakeholder Conflict</td><td style='padding:8px;'>5 actors with conflicting objectives and stochastic responses</td></tr>
                        <tr><td style='padding:8px; color:#ef4444;'>Economic Constraints</td><td style='padding:8px;'>Action costs deducted from limited budget; overspending penalized</td></tr>
                    </table>
                </div>""")

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
