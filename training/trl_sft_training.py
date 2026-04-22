import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from src.environment import DataCleaningEnv
from src.graders import (
    ComplexValidationGrader,
    DuplicateHandlingGrader,
    EnterpriseOrchestrationGrader,
    MissingValuesGrader,
)
from src.models import Action


OUTPUT_DIR = Path("artifacts")
POLICY_PATH = OUTPUT_DIR / "learned_policy.json"
POLICY_SNAPSHOTS_PATH = OUTPUT_DIR / "policy_snapshots.json"
METRICS_PATH = OUTPUT_DIR / "trl_sft_training_metrics.json"
CURVE_JSON_PATH = OUTPUT_DIR / "training_curve.json"
CURVE_CSV_PATH = OUTPUT_DIR / "training_curve.csv"
CURVE_SVG_PATH = OUTPUT_DIR / "training_curve.svg"

TASK_SPECS: Sequence[tuple[str, Callable]] = (
    ("task_missing_values", MissingValuesGrader),
    ("task_duplicate_handling", DuplicateHandlingGrader),
    ("task_complex_validation", ComplexValidationGrader),
    ("task_enterprise_orchestration", EnterpriseOrchestrationGrader),
)

ACTION_SPACE: Dict[str, List[Dict]] = {
    "task_missing_values": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "impute", "parameters": {"method": "forward_fill"}},
        {"action_type": "impute", "parameters": {"method": "mean"}},
        {"action_type": "deduplicate", "parameters": {"keep": "first"}},
        {"action_type": "validate", "parameters": {}},
        {
            "action_type": "report_findings",
            "parameters": {
                "include_summary": True,
                "include_quality_score": True,
                "include_recommendations": True,
            },
        },
    ],
    "task_duplicate_handling": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "deduplicate", "parameters": {"keep": "first"}},
        {"action_type": "deduplicate", "parameters": {"keep": "last"}},
        {"action_type": "validate", "parameters": {}},
        {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
        {
            "action_type": "report_findings",
            "parameters": {
                "include_summary": True,
                "include_quality_score": True,
                "include_recommendations": True,
            },
        },
    ],
    "task_complex_validation": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "impute", "parameters": {"method": "forward_fill"}},
        {"action_type": "deduplicate", "parameters": {"keep": "first"}},
        {"action_type": "validate", "parameters": {}},
        {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
        {
            "action_type": "report_findings",
            "parameters": {
                "include_summary": True,
                "include_quality_score": True,
                "include_recommendations": True,
            },
        },
    ],
    "task_enterprise_orchestration": [
        {"action_type": "analyze", "parameters": {}},
        {"action_type": "delegate", "parameters": {"actor": "finance_bot", "objective": "invoice cleanup"}},
        {"action_type": "delegate", "parameters": {"actor": "support_lead", "objective": "critical ticket triage"}},
        {"action_type": "resolve_alert", "parameters": {"actor": "finance_bot"}},
        {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}},
        {
            "action_type": "validate",
            "parameters": {
                "compliance_tier_type": "categorical_nonempty",
                "ticket_priority_type": "categorical_nonempty",
            },
        },
        {
            "action_type": "report_findings",
            "parameters": {
                "include_summary": True,
                "include_quality_score": True,
                "include_recommendations": True,
            },
        },
    ],
}


def _make_action(proto: Dict, columns: List[str], task_id: str) -> Action:
    if task_id == "task_enterprise_orchestration":
        targets = columns[:6]
    elif task_id == "task_duplicate_handling":
        targets = columns[:3]
    else:
        targets = columns[:4]
    return Action(
        action_type=proto["action_type"],
        target_columns=targets,
        parameters=dict(proto["parameters"]),
        reasoning=f"Training policy chose {proto['action_type']}.",
    )


def _evaluate_policy(policy: Dict[str, List[Dict]], episodes_per_task: int, seed_offset: int = 0) -> Dict:
    env = DataCleaningEnv(seed=1000 + seed_offset)
    task_scores: Dict[str, List[float]] = {task_id: [] for task_id, _ in TASK_SPECS}
    episode_records: List[Dict] = []

    for task_index, (task_id, grader) in enumerate(TASK_SPECS):
        for episode in range(episodes_per_task):
            seed = 4000 + seed_offset * 100 + task_index * 20 + episode
            observation = env.reset(task_id=task_id, seed=seed)
            actions = policy.get(task_id, [])
            total_reward = 0.0
            steps = 0
            for action_proto in actions:
                action = _make_action(action_proto, observation.column_names, task_id)
                observation, reward, done, _ = env.step(action)
                total_reward += reward.value
                steps += 1
                if done:
                    break
            grade = float(grader.grade(env.current_episode))
            task_scores[task_id].append(grade)
            episode_records.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "steps": steps,
                    "episode_reward": round(total_reward, 6),
                    "grade": round(grade, 6),
                }
            )

    per_task_means = {task_id: float(np.mean(scores)) for task_id, scores in task_scores.items()}
    all_scores = [score for scores in task_scores.values() for score in scores]
    return {
        "average_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores)),
        "task_scores": per_task_means,
        "episodes": episode_records,
    }


def _random_policy(rng: random.Random) -> Dict[str, List[Dict]]:
    policy: Dict[str, List[Dict]] = {}
    for task_id, _ in TASK_SPECS:
        options = ACTION_SPACE[task_id]
        sequence = []
        for _ in range(6):
            sequence.append(dict(rng.choice(options)))
        sequence[-1] = {
            "action_type": "report_findings",
            "parameters": {
                "include_summary": True,
                "include_quality_score": True,
                "include_recommendations": True,
            },
        }
        policy[task_id] = sequence
    return policy


def _mutate_policy(policy: Dict[str, List[Dict]], rng: random.Random) -> Dict[str, List[Dict]]:
    candidate = {task: [dict(action) for action in actions] for task, actions in policy.items()}
    task_id = rng.choice(list(candidate.keys()))
    options = ACTION_SPACE[task_id]
    idx = rng.randrange(len(candidate[task_id]))
    candidate[task_id][idx] = dict(rng.choice(options))
    if idx == len(candidate[task_id]) - 1:
        candidate[task_id][idx] = {
            "action_type": "report_findings",
            "parameters": {
                "include_summary": True,
                "include_quality_score": True,
                "include_recommendations": True,
            },
        }
    return candidate


def _write_training_curve(curve: List[Dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_JSON_PATH.write_text(json.dumps(curve, indent=2), encoding="utf-8")

    with CURVE_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        f.write("iteration,average_score,best_score\n")
        for row in curve:
            f.write(f"{row['iteration']},{row['average_score']:.6f},{row['best_score']:.6f}\n")

    width = 720
    height = 320
    left = 70
    top = 25
    plot_w = 620
    plot_h = 230
    values = [row["average_score"] for row in curve]
    best_values = [row["best_score"] for row in curve]
    min_v = min(values + best_values) - 0.02
    max_v = max(values + best_values) + 0.02
    span = max(max_v - min_v, 1e-6)

    def _points(series: List[float]) -> str:
        points: List[str] = []
        for i, value in enumerate(series):
            x = left + int((i / max(len(series) - 1, 1)) * plot_w)
            y = top + int(((max_v - value) / span) * plot_h)
            points.append(f"{x},{y}")
        return " ".join(points)

    avg_points = _points(values)
    best_points = _points(best_values)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="16" font-size="14" font-family="Arial">Environment-Grounded Training Curve</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{avg_points}"/>',
        f'<polyline fill="none" stroke="#059669" stroke-width="2.5" points="{best_points}"/>',
        f'<text x="{left + 20}" y="{top + 20}" font-size="11" font-family="Arial" fill="#2563eb">candidate score</text>',
        f'<text x="{left + 20}" y="{top + 36}" font-size="11" font-family="Arial" fill="#059669">best-so-far score</text>',
        f'<text x="{left + plot_w - 160}" y="{top + plot_h + 20}" font-size="11" font-family="Arial">x-axis: training iteration</text>',
        f'<text x="{left + 6}" y="{top + 12}" font-size="11" font-family="Arial">y: average reward/grade</text>',
        "</svg>",
    ]
    CURVE_SVG_PATH.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)

    baseline_policy = _random_policy(rng)
    baseline_eval = _evaluate_policy(baseline_policy, episodes_per_task=6, seed_offset=0)

    best_policy = baseline_policy
    best_eval = baseline_eval
    mid_policy_snapshot = {task: [dict(action) for action in actions] for task, actions in best_policy.items()}
    trajectory: List[Dict] = []
    iterations = 60
    accepted = 0

    for iteration in range(1, iterations + 1):
        candidate_policy = _mutate_policy(best_policy, rng)
        candidate_eval = _evaluate_policy(candidate_policy, episodes_per_task=4, seed_offset=iteration)

        if candidate_eval["average_score"] > best_eval["average_score"]:
            best_policy = candidate_policy
            best_eval = candidate_eval
            accepted += 1

        if iteration == iterations // 2:
            mid_policy_snapshot = {task: [dict(action) for action in actions] for task, actions in best_policy.items()}

        trajectory.append(
            {
                "iteration": iteration,
                "average_score": round(candidate_eval["average_score"], 6),
                "best_score": round(best_eval["average_score"], 6),
            }
        )

    final_eval = _evaluate_policy(best_policy, episodes_per_task=10, seed_offset=999)
    random_eval = _evaluate_policy(_random_policy(random.Random(9090)), episodes_per_task=10, seed_offset=555)

    per_task_improvement = {}
    for task_id, _ in TASK_SPECS:
        base = random_eval["task_scores"][task_id]
        final = final_eval["task_scores"][task_id]
        per_task_improvement[task_id] = round(final - base, 6)

    POLICY_PATH.write_text(json.dumps(best_policy, indent=2), encoding="utf-8")
    snapshots = {
        "baseline": baseline_policy,
        "mid": mid_policy_snapshot,
        "trained": best_policy,
    }
    POLICY_SNAPSHOTS_PATH.write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    _write_training_curve(trajectory)

    action_hist = Counter(
        action["action_type"]
        for task_actions in best_policy.values()
        for action in task_actions
    )
    metrics = {
        "training_mode": "environment_grounded_policy_search",
        "optimizer": "hill_climbing_over_action_programs",
        "episodes_per_task_baseline": 6,
        "episodes_per_task_eval": 10,
        "iterations": iterations,
        "accepted_updates": accepted,
        "baseline_average_score": round(random_eval["average_score"], 6),
        "trained_average_score": round(final_eval["average_score"], 6),
        "improvement": round(final_eval["average_score"] - random_eval["average_score"], 6),
        "baseline_std": round(random_eval["std_score"], 6),
        "trained_std": round(final_eval["std_score"], 6),
        "task_scores_baseline": {k: round(v, 6) for k, v in random_eval["task_scores"].items()},
        "task_scores_trained": {k: round(v, 6) for k, v in final_eval["task_scores"].items()},
        "task_improvement": per_task_improvement,
        "action_distribution": dict(action_hist),
        "artifacts": {
            "policy": str(POLICY_PATH),
            "policy_snapshots": str(POLICY_SNAPSHOTS_PATH),
            "training_curve_json": str(CURVE_JSON_PATH),
            "training_curve_csv": str(CURVE_CSV_PATH),
            "training_curve_svg": str(CURVE_SVG_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
