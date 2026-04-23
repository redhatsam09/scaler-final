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
        from pathlib import Path
    except ImportError:
        return None

    demo_env = DataCleaningEnv(seed=42)
    demo_session = {"obs": None, "history": [], "task_id": "task_enterprise_orchestration"}

    def reset_env(task_id, difficulty, seed):
        seed_val = int(seed) if seed else 42
        obs = demo_env.reset(task_id=task_id, seed=seed_val, difficulty=difficulty)
        demo_session["obs"] = obs
        demo_session["history"] = []
        demo_session["task_id"] = task_id

        state_text = obs.natural_language_observation
        kpi_text = "\n".join(f"- {k}: {v:.3f}" for k, v in obs.kpi_snapshot.items())
        urgency = "\n".join(f"- ALERT: {s}" for s in obs.urgency_signals) if obs.urgency_signals else "- No active alerts"
        actors = "\n".join(f"- MESSAGE: {m}" for m in obs.actor_messages) if obs.actor_messages else "- No active communications"

        output = f"""### System Initialized
**Task:** {task_id} | **Difficulty:** {difficulty} | **Seed:** {seed_val} | **Dataset:** {obs.dataset_shape[0]} rows × {obs.dataset_shape[1]} cols

**Observation State:**
{state_text}

**Key Performance Indicators:**
{kpi_text}

**Urgency Signals:**
{urgency}

**Actor Communications:**
{actors}

**Available Actions:** {', '.join(obs.available_actions)}
"""
        return output, ""

    def step_env(action_type, target_cols, params_json, reasoning):
        if demo_session["obs"] is None:
            return "ERROR: System not initialized. Please reset the environment first.", ""

        try:
            params = json.loads(params_json) if params_json.strip() else {}
        except json.JSONDecodeError:
            params = {}

        cols = [c.strip() for c in target_cols.split(",") if c.strip()] if target_cols else []
        action = Action(action_type=action_type, target_columns=cols, parameters=params, reasoning=reasoning or "System execution")
        obs, reward, done, info = demo_env.step(action)
        demo_session["obs"] = obs
        demo_session["history"].append({"action": action_type, "reward": reward.value})

        kpi_text = "\n".join(f"- {k}: {v:.3f}" for k, v in obs.kpi_snapshot.items())
        urgency = "\n".join(f"- ALERT: {s}" for s in obs.urgency_signals) if obs.urgency_signals else "- No active alerts"
        actors = "\n".join(f"- MESSAGE: {m}" for m in obs.actor_messages[-3:]) if obs.actor_messages else "- No active communications"
        components = "\n".join(f"- {k}: {v:.4f}" for k, v in info.get("components", {}).items())

        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
            "task_enterprise_orchestration": EnterpriseOrchestrationGrader,
        }
        grade = graders[demo_session["task_id"]].grade(demo_env.current_episode)

        history_text = " → ".join(f"{h['action']}({h['reward']:.2f})" for h in demo_session["history"][-6:])

        output = f"""### {'EPISODE TERMINATED' if done else f'Execution Step: {obs.step_count}'}
**Action Processed:** {action_type} | **Step Reward:** {reward.value:.4f} | **Cumulative Grade:** {grade:.4f} | **Status:** {'Done' if done else 'Active'}

**Observation State:**
{obs.natural_language_observation}

**Key Performance Indicators:**
{kpi_text}

**Reward Decomposition:**
{components}

**Urgency Signals:**
{urgency}

**Actor Communications:**
{actors}

**Execution History:** {history_text}
"""
        return output, ""

    custom_css = """
    .gradio-container { font-family: 'Inter', sans-serif; }
    h1 { color: #1e293b; text-align: left; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
    .gr-button-primary { background-color: #0f172a; color: white; font-weight: 600; border-radius: 4px; }
    .gr-button-secondary { background-color: #334155; color: white; font-weight: 600; border-radius: 4px; }
    .dark .gr-button-primary { background-color: #3b82f6; }
    .dark .gr-button-secondary { background-color: #64748b; }
    """

    with gr.Blocks(title="Enterprise Orchestration Console") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        gr.Markdown("# Enterprise Orchestration Console")
        gr.Markdown("*Professional LLM Reinforcement Learning Environment (Theme: World Modeling & Multi-Agent)*")

        with gr.Tabs():
            with gr.TabItem("Environment Simulator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        task_dd = gr.Dropdown(
                            choices=["task_enterprise_orchestration", "task_missing_values",
                                     "task_duplicate_handling", "task_complex_validation"],
                            value="task_enterprise_orchestration", label="Task Identifier"
                        )
                        diff_dd = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Simulation Difficulty")
                        seed_tb = gr.Textbox(value="42", label="Random Seed")
                        reset_btn = gr.Button("Initialize System", variant="primary")

                        gr.Markdown("### Execution Panel")
                        action_dd = gr.Dropdown(
                            choices=["analyze", "impute", "deduplicate", "validate", "report_findings",
                                     "delegate", "resolve_alert", "reconcile_apps", "oversight_review",
                                     "inspect_actor", "audit_records", "request_policy_clarification"],
                            value="analyze", label="Action Type"
                        )
                        cols_tb = gr.Textbox(label="Target Columns (comma-separated)", placeholder="account_id, invoice_status")
                        params_tb = gr.Textbox(label="Action Parameters (JSON)", value="{}", lines=2)
                        reason_tb = gr.Textbox(label="Strategic Reasoning", placeholder="Provide rationale for model execution...")
                        step_btn = gr.Button("Execute Step", variant="secondary")

                    with gr.Column(scale=2):
                        output_md = gr.Markdown("*Awaiting system initialization...*")
                        error_md = gr.Markdown("")

                reset_btn.click(reset_env, inputs=[task_dd, diff_dd, seed_tb], outputs=[output_md, error_md])
                step_btn.click(step_env, inputs=[action_dd, cols_tb, params_tb, reason_tb], outputs=[output_md, error_md])

            with gr.TabItem("Training Evidence (GRPO)"):
                gr.Markdown("""
                ### Reinforcement Learning Performance Metrics
                The following artifacts demonstrate the progressive capability improvement of an LLM agent trained on this environment using Generative Reward Policy Optimization (GRPO).
                """)
                with gr.Row():
                    try:
                        import os
                        tc_path = "artifacts/training_curve.svg"
                        rp_path = "artifacts/reward_progression.svg"
                        
                        if not os.path.exists(tc_path):
                            # Fallback if running from a different working directory
                            tc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), tc_path)
                            rp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rp_path)

                        gr.Image(value=tc_path, type="filepath", label="Training Curve", show_download_button=False)
                        gr.Image(value=rp_path, type="filepath", label="Reward Progression", show_download_button=False)
                    except Exception as e:
                        gr.Markdown(f"*(Error loading artifacts: {e}. Please run `training/evaluate_reward_improvement.py` to generate the plots.)*")

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
