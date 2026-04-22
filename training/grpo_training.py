"""
Real GRPO Training Script for Enterprise Orchestration Environment.

Uses TRL GRPOTrainer + Unsloth for efficient RL training on the
actual environment with verifiable rewards.

Run: python training/grpo_training.py
Or use in Colab with the provided notebook.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = Path("artifacts")
GRPO_METRICS_PATH = OUTPUT_DIR / "grpo_training_metrics.json"

# --- Environment reward function ---
from src.environment import DataCleaningEnv
from src.graders import EnterpriseOrchestrationGrader, MissingValuesGrader
from src.models import Action

SYSTEM_PROMPT = """You are an enterprise workflow orchestrator managing CRM, Billing, and Support systems.

Given the current state, output a JSON action with:
- action_type: one of [analyze, impute, deduplicate, validate, report_findings, delegate, resolve_alert, reconcile_apps, oversight_review, inspect_actor, audit_records, request_policy_clarification]
- target_columns: list of column names
- parameters: dict of action parameters
- reasoning: why you chose this action

Respond ONLY with valid JSON."""


def build_prompt(observation) -> str:
    """Convert observation to training prompt."""
    return f"""{SYSTEM_PROMPT}

Current State:
{observation.natural_language_observation}

Dataset: {observation.dataset_shape[0]} rows, {observation.dataset_shape[1]} cols
Missing values: {json.dumps(dict(list(observation.missing_values.items())[:6]))}
Step: {observation.step_count}
Policy version: {observation.policy_version}
KPIs: {json.dumps(observation.kpi_snapshot)}
Actor messages: {observation.actor_messages[-2:] if observation.actor_messages else []}
Urgency: {observation.urgency_signals if observation.urgency_signals else 'None'}

What action should you take?"""


def parse_action_from_text(text: str, columns: list) -> Action:
    """Parse LLM output into Action, with fallback."""
    import re
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return Action(
                action_type=data.get("action_type", "analyze"),
                target_columns=data.get("target_columns", columns[:3]),
                parameters=data.get("parameters", {}),
                reasoning=data.get("reasoning", "Model-generated action"),
            )
    except (json.JSONDecodeError, KeyError):
        pass
    return Action(
        action_type="analyze",
        target_columns=columns[:3],
        parameters={},
        reasoning="Fallback: could not parse model output",
    )


def environment_reward_function(completions: list, prompts: list, **kwargs) -> list:
    """
    Reward function that runs each completion through the actual environment.
    This is the core of GRPO — verifiable rewards from the environment.
    """
    rewards = []
    env = DataCleaningEnv(seed=42)
    task_ids = ["task_enterprise_orchestration", "task_missing_values"]

    for i, completion in enumerate(completions):
        try:
            task_id = task_ids[i % len(task_ids)]
            obs = env.reset(task_id=task_id, seed=1000 + i)
            action = parse_action_from_text(completion, obs.column_names)
            _, reward, _, info = env.step(action)

            # Multi-component reward
            env_reward = reward.value
            grade = 0.0
            if task_id == "task_enterprise_orchestration":
                grade = EnterpriseOrchestrationGrader.grade(env.current_episode)
            else:
                grade = MissingValuesGrader.grade(env.current_episode)

            # Combined: step reward + partial grade
            combined = 0.6 * env_reward + 0.4 * grade

            # Bonus for valid JSON
            try:
                import re
                match = re.search(r'\{.*\}', completion, re.DOTALL)
                if match:
                    json.loads(match.group())
                    combined += 0.1
            except Exception:
                combined -= 0.1

            rewards.append(min(1.0, max(-1.0, combined)))
        except Exception:
            rewards.append(-0.5)

    return rewards


def main():
    """
    Main GRPO training entrypoint.

    When TRL + Unsloth are available (Colab/GPU), runs real GRPO training.
    Otherwise, generates training data and metrics from environment rollouts.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Check if TRL + Unsloth are available ---
    trl_available = False
    unsloth_available = False
    try:
        from trl import GRPOTrainer, GRPOConfig
        trl_available = True
    except ImportError:
        pass
    try:
        from unsloth import FastLanguageModel
        unsloth_available = True
    except ImportError:
        pass

    print(f"TRL available: {trl_available}")
    print(f"Unsloth available: {unsloth_available}")

    if trl_available and unsloth_available:
        print("=" * 60)
        print("RUNNING REAL GRPO TRAINING WITH UNSLOTH")
        print("=" * 60)
        _run_grpo_training()
    elif trl_available:
        print("=" * 60)
        print("RUNNING GRPO TRAINING (without Unsloth optimization)")
        print("=" * 60)
        _run_grpo_training()
    else:
        print("=" * 60)
        print("TRL not available — generating training data + rollout metrics")
        print("This script is designed for Colab with GPU. See notebook.")
        print("=" * 60)
        _generate_training_data_and_metrics()


def _run_grpo_training():
    """Real GRPO training with TRL and optional Unsloth."""
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM

    try:
        from unsloth import FastLanguageModel
        model_name = "unsloth/Qwen2.5-1.5B-Instruct"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )
        print(f"Loaded {model_name} with Unsloth 4-bit + LoRA")
    except ImportError:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded {model_name} without Unsloth")

    # Generate prompts from environment
    env = DataCleaningEnv(seed=42)
    prompts = []
    for i in range(50):
        task_id = ["task_enterprise_orchestration", "task_missing_values",
                   "task_duplicate_handling", "task_complex_validation"][i % 4]
        obs = env.reset(task_id=task_id, seed=2000 + i)
        prompts.append(build_prompt(obs))

    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR / "grpo_checkpoint"),
        num_generations=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=30,
        logging_steps=1,
        save_strategy="no",
        max_completion_length=512,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_funcs=[environment_reward_function],
    )

    result = trainer.train()

    metrics = {
        "training_mode": "grpo_rl_training",
        "model": model_name,
        "unsloth_used": "unsloth" in model_name.lower(),
        "trl_used": True,
        "num_prompts": len(prompts),
        "train_loss": float(result.training_loss) if hasattr(result, "training_loss") else None,
        "max_steps": 30,
        "num_generations": 4,
    }
    GRPO_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"GRPO training complete. Metrics saved to {GRPO_METRICS_PATH}")


def _generate_training_data_and_metrics():
    """
    When TRL is not available, generate training data from environment rollouts.
    This produces the dataset that would be used for GRPO training in Colab.
    """
    import random

    env = DataCleaningEnv(seed=42)
    training_examples = []
    rollout_rewards = []

    tasks = [
        "task_enterprise_orchestration",
        "task_missing_values",
        "task_duplicate_handling",
        "task_complex_validation",
    ]

    # Expert action sequences for each task
    expert_sequences = {
        "task_enterprise_orchestration": [
            {"action_type": "analyze", "parameters": {}},
            {"action_type": "inspect_actor", "parameters": {"actor": "finance_bot"}},
            {"action_type": "inspect_actor", "parameters": {"actor": "analytics_assistant"}},
            {"action_type": "delegate", "parameters": {"actor": "finance_bot", "objective": "invoice cleanup"}},
            {"action_type": "resolve_alert", "parameters": {"actor": "finance_bot"}},
            {"action_type": "oversight_review", "parameters": {"actor": "analytics_assistant", "explain": True}},
            {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
            {"action_type": "validate", "parameters": {"compliance_tier_type": "categorical_nonempty"}},
            {"action_type": "report_findings", "parameters": {"include_summary": True, "include_quality_score": True,
                                                              "include_recommendations": True, "include_actor_tradeoffs": True,
                                                              "include_budget_analysis": True}},
        ],
        "task_missing_values": [
            {"action_type": "analyze", "parameters": {}},
            {"action_type": "impute", "parameters": {"method": "forward_fill"}},
            {"action_type": "impute", "parameters": {"method": "mean"}},
            {"action_type": "validate", "parameters": {}},
            {"action_type": "report_findings", "parameters": {"include_summary": True, "include_quality_score": True,
                                                              "include_recommendations": True}},
        ],
        "task_duplicate_handling": [
            {"action_type": "analyze", "parameters": {}},
            {"action_type": "deduplicate", "parameters": {"subset": ["invoice_id"], "keep": "first"}},
            {"action_type": "validate", "parameters": {}},
            {"action_type": "report_findings", "parameters": {"include_summary": True, "include_quality_score": True,
                                                              "include_recommendations": True}},
        ],
        "task_complex_validation": [
            {"action_type": "analyze", "parameters": {}},
            {"action_type": "impute", "parameters": {"method": "forward_fill"}},
            {"action_type": "deduplicate", "parameters": {"keep": "first"}},
            {"action_type": "validate", "parameters": {}},
            {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
            {"action_type": "report_findings", "parameters": {"include_summary": True, "include_quality_score": True,
                                                              "include_recommendations": True}},
        ],
    }

    for seed_offset in range(20):
        for task_id in tasks:
            seed = 3000 + seed_offset * 10
            obs = env.reset(task_id=task_id, seed=seed)
            prompt = build_prompt(obs)
            total_reward = 0.0

            for action_proto in expert_sequences[task_id]:
                action = Action(
                    action_type=action_proto["action_type"],
                    target_columns=obs.column_names[:4],
                    parameters=action_proto["parameters"],
                    reasoning=f"Expert policy: {action_proto['action_type']} for {task_id}",
                )
                obs, reward, done, _ = env.step(action)
                total_reward += reward.value
                if done:
                    break

            completion = json.dumps({
                "action_type": expert_sequences[task_id][0]["action_type"],
                "target_columns": obs.column_names[:3],
                "parameters": expert_sequences[task_id][0]["parameters"],
                "reasoning": f"Expert strategy for {task_id}",
            })

            training_examples.append({
                "prompt": prompt,
                "completion": completion,
                "task_id": task_id,
                "seed": seed,
                "total_reward": round(total_reward, 4),
            })
            rollout_rewards.append(total_reward)

    # Save training dataset
    dataset_path = OUTPUT_DIR / "grpo_training_dataset.json"
    dataset_path.write_text(json.dumps(training_examples, indent=2))

    metrics = {
        "training_mode": "grpo_training_data_generation",
        "note": "Run this script in Colab with GPU for actual GRPO training",
        "trl_required": True,
        "unsloth_recommended": True,
        "model_recommended": "unsloth/Qwen2.5-1.5B-Instruct",
        "num_training_examples": len(training_examples),
        "mean_reward": round(float(sum(rollout_rewards) / len(rollout_rewards)), 4),
        "max_reward": round(float(max(rollout_rewards)), 4),
        "min_reward": round(float(min(rollout_rewards)), 4),
        "tasks_covered": tasks,
        "dataset_path": str(dataset_path),
    }
    GRPO_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Generated {len(training_examples)} training examples")
    print(f"Mean reward: {metrics['mean_reward']}")
    print(f"Saved to: {dataset_path}")
    print(f"Metrics: {GRPO_METRICS_PATH}")


if __name__ == "__main__":
    main()
