import os
import json
import re
from typing import Optional
from openai import OpenAI
from src.environment import DataCleaningEnv
from src.models import Action
from src.graders import MissingValuesGrader, DuplicateHandlingGrader, ComplexValidationGrader

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("MODEL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "auto").strip().lower()
MAX_STEPS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 500
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99
INFERENCE_SEED = int(os.getenv("INFERENCE_SEED", "2026"))

SYSTEM_PROMPT = """You are an expert data analyst. Your task is to clean and validate datasets by identifying and resolving data quality issues.

You have access to the following actions:
- analyze: Examine data structure, types, and quality metrics
- impute: Fill missing values using specified strategy (mean, median, forward_fill)
- deduplicate: Remove duplicate records
- validate: Check data against validation rules
- report_findings: Generate summary of data quality assessment

For each action, provide:
1. action_type: One of the actions above
2. target_columns: List of column names to process
3. parameters: Dictionary with method-specific parameters
4. reasoning: Brief explanation of your approach

Format your response as valid JSON."""


def emit(event: str, **fields) -> None:
    parts = [f"[{event}]"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def bound_task_score(value: float) -> float:
    if value <= MIN_TASK_SCORE:
        return MIN_TASK_SCORE
    if value >= MAX_TASK_SCORE:
        return MAX_TASK_SCORE
    return float(value)


def extract_action(response_text: str) -> Optional[Action]:
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return None
        
        action_data = json.loads(json_match.group())
        
        return Action(
            action_type=action_data.get('action_type', 'analyze'),
            target_columns=action_data.get('target_columns', []),
            parameters=action_data.get('parameters', {}),
            reasoning=action_data.get('reasoning', '')
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def should_use_remote_model() -> bool:
    if INFERENCE_BACKEND == "local":
        return False
    if INFERENCE_BACKEND == "api":
        return bool(API_KEY)
    return bool(API_KEY)


def local_policy_action(task_id: str, observation, step: int) -> Action:
    columns = observation.column_names
    missing_columns = [
        column
        for column, missing_count in observation.missing_values.items()
        if missing_count > 0
    ]
    working_columns = missing_columns or columns

    if step == 1:
        return Action(
            action_type="analyze",
            target_columns=working_columns,
            parameters={},
            reasoning="Local deterministic policy: profile columns with active quality issues.",
        )

    if task_id == "task_duplicate_handling":
        if step == 2:
            subset = [columns[0]] if columns else None
            return Action(
                action_type="deduplicate",
                target_columns=columns,
                parameters={"subset": subset, "keep": "first"},
                reasoning="Local deterministic policy: remove duplicate invoice records by primary identifier.",
            )
        if step == 3:
            return Action(
                action_type="validate",
                target_columns=columns[:2] or columns,
                parameters={},
                reasoning="Local deterministic policy: validate key invoice fields after deduplication.",
            )

    else:
        if step == 2:
            return Action(
                action_type="impute",
                target_columns=working_columns,
                parameters={"method": "forward_fill"},
                reasoning="Local deterministic policy: fill missing values without external model calls.",
            )
        if step == 3:
            return Action(
                action_type="deduplicate",
                target_columns=columns,
                parameters={"keep": "first"},
                reasoning="Local deterministic policy: remove exact duplicate records.",
            )
        if step == 4:
            return Action(
                action_type="validate",
                target_columns=columns,
                parameters={},
                reasoning="Local deterministic policy: validate cleaned dataset columns.",
            )

    return Action(
        action_type="report_findings",
        target_columns=columns[:1],
        parameters={
            "include_summary": True,
            "include_quality_score": True,
            "include_recommendations": True,
        },
        reasoning="Local deterministic policy: summarize cleaning results.",
    )


def run_task(env: DataCleaningEnv, task_id: str, grader_class, seed: int) -> float:
    client = None
    if should_use_remote_model():
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    observation = env.reset(task_id=task_id, seed=seed)
    emit("START", task=task_id, seed=seed)
    steps_executed = 0
    
    for step in range(1, MAX_STEPS + 1):
        state_description = f"""
Current dataset state:
- Shape: {observation.dataset_shape}
- Columns: {', '.join(observation.column_names)}
- Data types: {json.dumps(observation.data_types, indent=2)}
- Missing values: {json.dumps(observation.missing_values, indent=2)}
- Current progress: {observation.episode_progress}

Based on this state, what data cleaning action should you take next?
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_description}
        ]
        
        response_text = ""
        if client is not None:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception:
                emit("STEP", task=task_id, step=step, action="api_error", reward="0.000000", done="false")
        
        action = extract_action(response_text)
        if not action:
            action = local_policy_action(task_id, observation, step)
        
        observation, reward, done, info = env.step(action)
        steps_executed = step
        emit(
            "STEP",
            task=task_id,
            step=step,
            action=action.action_type,
            reward=f"{reward.value:.6f}",
            done=str(done).lower(),
        )
        
        if done:
            break
    
    episode_state = env.current_episode
    final_score = bound_task_score(grader_class.grade(episode_state))
    emit("END", task=task_id, score=f"{final_score:.6f}", steps=steps_executed)
    return final_score


def main():
    env = DataCleaningEnv()
    
    tasks = [
        ("task_missing_values", MissingValuesGrader),
        ("task_duplicate_handling", DuplicateHandlingGrader),
        ("task_complex_validation", ComplexValidationGrader),
    ]
    
    scores = {}
    
    for idx, (task_id, grader_class) in enumerate(tasks):
        try:
            score = run_task(env, task_id, grader_class, INFERENCE_SEED + idx)
            scores[task_id] = score
        except Exception:
            emit("START", task=task_id)
            emit("END", task=task_id, score="0.010000", steps=0)
            scores[task_id] = MIN_TASK_SCORE

    average_score = sum(scores.values()) / len(scores) if scores else 0.0
    emit("SUMMARY", average_score=f"{average_score:.6f}")
    
    return average_score


if __name__ == "__main__":
    main()
