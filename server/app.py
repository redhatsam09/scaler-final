import json
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from src.environment import DataCleaningEnv, DEFAULT_ENV_SEED
from src.models import Action
from src.graders import (
    MissingValuesGrader,
    DuplicateHandlingGrader,
    ComplexValidationGrader,
    EnterpriseOrchestrationGrader,
)

app = FastAPI(title="Enterprise Orchestration Environment")

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


SESSION_TTL_SECONDS = 3600


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(environments),
        "session_ttl_seconds": SESSION_TTL_SECONDS,
    }


@app.get("/")
async def root():
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

        env = environments.get(session_id)
        if env is None:
            env = DataCleaningEnv(seed=seed if seed is not None else DEFAULT_ENV_SEED)
            environments[session_id] = env

        observation = env.reset(task_id=task_id, seed=seed, difficulty=difficulty)

        return ResetResponse(
            session_id=session_id,
            observation=_obs_to_dict(observation),
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
            observation=_obs_to_dict(observation),
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
            "task_enterprise_orchestration": EnterpriseOrchestrationGrader,
        }

        grader_class = graders.get(task_id)
        if not grader_class:
            raise ValueError(f"Unknown task: {task_id}")

        score = grader_class.grade(env.current_episode)
        return GradeResponse(session_id=resolved_session_id, task_id=task_id, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close")
async def close(session_id: Optional[str] = None):
    """Close a session and free its resources."""
    try:
        resolved_session_id = _resolve_runtime_session(session_id)
        env = environments.pop(resolved_session_id, None)
        if env is None:
            raise ValueError(f"Unknown session_id: {resolved_session_id}")
        return {"status": "closed", "session_id": resolved_session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Gradio Interactive Demo ----

def _build_gradio_demo():
    """Build Gradio UI for judges to interact with the environment."""
    try:
        import gradio as gr
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
        kpi_text = "\n".join(f"  {k}: {v:.3f}" for k, v in obs.kpi_snapshot.items())
        urgency = "\n".join(f"  ⚠️ {s}" for s in obs.urgency_signals) if obs.urgency_signals else "  None"
        actors = "\n".join(f"  💬 {m}" for m in obs.actor_messages) if obs.actor_messages else "  None"

        output = f"""🔄 **Environment Reset**
**Task:** {task_id} | **Difficulty:** {difficulty} | **Seed:** {seed_val}
**Dataset:** {obs.dataset_shape[0]} rows × {obs.dataset_shape[1]} cols

📝 **Natural Language Observation:**
{state_text}

📊 **KPIs:**
{kpi_text}

⚠️ **Urgency Signals:**
{urgency}

💬 **Actor Messages:**
{actors}

**Available Actions:** {', '.join(obs.available_actions)}
"""
        return output, ""

    def step_env(action_type, target_cols, params_json, reasoning):
        if demo_session["obs"] is None:
            return "❌ Reset the environment first!", ""

        try:
            params = json.loads(params_json) if params_json.strip() else {}
        except json.JSONDecodeError:
            params = {}

        cols = [c.strip() for c in target_cols.split(",") if c.strip()] if target_cols else []
        action = Action(action_type=action_type, target_columns=cols, parameters=params, reasoning=reasoning or "Manual step")
        obs, reward, done, info = demo_env.step(action)
        demo_session["obs"] = obs
        demo_session["history"].append({"action": action_type, "reward": reward.value})

        kpi_text = "\n".join(f"  {k}: {v:.3f}" for k, v in obs.kpi_snapshot.items())
        urgency = "\n".join(f"  ⚠️ {s}" for s in obs.urgency_signals) if obs.urgency_signals else "  None"
        actors = "\n".join(f"  💬 {m}" for m in obs.actor_messages[-3:]) if obs.actor_messages else "  None"
        components = "\n".join(f"  {k}: {v:.4f}" for k, v in info.get("components", {}).items())

        # Get grade
        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
            "task_enterprise_orchestration": EnterpriseOrchestrationGrader,
        }
        grade = graders[demo_session["task_id"]].grade(demo_env.current_episode)

        history_text = " → ".join(f"{h['action']}({h['reward']:.2f})" for h in demo_session["history"][-6:])

        output = f"""{'🏁 EPISODE COMPLETE' if done else f'Step {obs.step_count}'}
**Action:** {action_type} | **Reward:** {reward.value:.4f} | **Grade:** {grade:.4f} | **Done:** {done}

📝 **Observation:**
{obs.natural_language_observation}

📊 **KPIs:**
{kpi_text}

🧮 **Reward Components:**
{components}

⚠️ **Urgency:**
{urgency}

💬 **Actor Messages:**
{actors}

📈 **History:** {history_text}
"""
        return output, ""

    with gr.Blocks(title="Enterprise Orchestration Environment", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""# 🏢 Enterprise Orchestration Environment
*Multi-app RL environment with schema drift, actor conflicts, deceptive oversight, and economic budgets*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔄 Reset")
                task_dd = gr.Dropdown(
                    choices=["task_enterprise_orchestration", "task_missing_values",
                             "task_duplicate_handling", "task_complex_validation"],
                    value="task_enterprise_orchestration", label="Task"
                )
                diff_dd = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Difficulty")
                seed_tb = gr.Textbox(value="42", label="Seed")
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

                gr.Markdown("### 🎮 Step")
                action_dd = gr.Dropdown(
                    choices=["analyze", "impute", "deduplicate", "validate", "report_findings",
                             "delegate", "resolve_alert", "reconcile_apps", "oversight_review",
                             "inspect_actor", "audit_records", "request_policy_clarification"],
                    value="analyze", label="Action"
                )
                cols_tb = gr.Textbox(label="Target Columns (comma-separated)", placeholder="account_id, invoice_status")
                params_tb = gr.Textbox(label="Parameters (JSON)", value="{}", lines=2)
                reason_tb = gr.Textbox(label="Reasoning", placeholder="Why this action?")
                step_btn = gr.Button("▶️ Execute Step", variant="secondary")

            with gr.Column(scale=2):
                output_md = gr.Markdown("*Reset the environment to begin*")
                error_md = gr.Markdown("")

        reset_btn.click(reset_env, inputs=[task_dd, diff_dd, seed_tb], outputs=[output_md, error_md])
        step_btn.click(step_env, inputs=[action_dd, cols_tb, params_tb, reason_tb], outputs=[output_md, error_md])

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
