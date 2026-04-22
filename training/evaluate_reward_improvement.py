import csv
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.environment import DataCleaningEnv
from src.graders import (
    ComplexValidationGrader,
    DuplicateHandlingGrader,
    EnterpriseOrchestrationGrader,
    MissingValuesGrader,
)
from src.models import Action


OUTPUT_DIR = Path("artifacts")
CSV_PATH = OUTPUT_DIR / "reward_progression.csv"
JSON_PATH = OUTPUT_DIR / "reward_progression.json"
SVG_PATH = OUTPUT_DIR / "reward_progression.svg"
POLICY_PATH = OUTPUT_DIR / "learned_policy.json"
POLICY_SNAPSHOTS_PATH = OUTPUT_DIR / "policy_snapshots.json"
METRICS_PATH = OUTPUT_DIR / "trl_sft_training_metrics.json"


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


def _load_policy() -> Dict[str, List[Dict]]:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Missing learned policy at {POLICY_PATH}. Run training/trl_sft_training.py first.")
    with POLICY_PATH.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return {task_id: [dict(action) for action in actions] for task_id, actions in loaded.items()}


def _load_policy_snapshots() -> Dict[str, Dict[str, List[Dict]]]:
    if not POLICY_SNAPSHOTS_PATH.exists():
        trained = _load_policy()
        return {"baseline": _random_policy(seed=2026), "mid": trained, "trained": trained}
    with POLICY_SNAPSHOTS_PATH.open("r", encoding="utf-8") as f:
        snapshots = json.load(f)
    normalized: Dict[str, Dict[str, List[Dict]]] = {}
    for stage, policy in snapshots.items():
        normalized[stage] = {task_id: [dict(action) for action in actions] for task_id, actions in policy.items()}
    if "trained" not in normalized:
        normalized["trained"] = _load_policy()
    if "mid" not in normalized:
        normalized["mid"] = normalized["trained"]
    if "baseline" not in normalized:
        normalized["baseline"] = _random_policy(seed=2026)
    return normalized


def _random_policy(seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    policy: Dict[str, List[Dict]] = {}
    for task_id, _ in TASK_SPECS:
        options = ACTION_SPACE[task_id]
        sequence = [dict(rng.choice(options)) for _ in range(6)]
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


def _action_from_proto(task_id: str, proto: Dict, columns: List[str]) -> Action:
    if task_id == "task_enterprise_orchestration":
        target_cols = columns[:6]
    elif task_id == "task_duplicate_handling":
        target_cols = columns[:3]
    else:
        target_cols = columns[:4]
    return Action(
        action_type=proto["action_type"],
        target_columns=target_cols,
        parameters=dict(proto["parameters"]),
        reasoning=f"Evaluation policy chose {proto['action_type']}.",
    )


def _evaluate_policy(
    policy: Dict[str, List[Dict]],
    stage: str,
    episodes_per_task: int,
    base_seed: int,
) -> Dict:
    env = DataCleaningEnv(seed=base_seed)
    task_scores: Dict[str, List[float]] = {task_id: [] for task_id, _ in TASK_SPECS}
    for task_idx, (task_id, grader) in enumerate(TASK_SPECS):
        for episode_idx in range(episodes_per_task):
            seed = base_seed + task_idx * 100 + episode_idx
            observation = env.reset(task_id=task_id, seed=seed)
            for proto in policy.get(task_id, []):
                action = _action_from_proto(task_id, proto, observation.column_names)
                observation, _, done, _ = env.step(action)
                if done:
                    break
            score = float(grader.grade(env.current_episode))
            task_scores[task_id].append(score)

    per_task_mean = {task_id: float(np.mean(values)) for task_id, values in task_scores.items()}
    all_scores = [score for scores in task_scores.values() for score in scores]
    return {
        "stage": stage,
        "average_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores)),
        "task_scores": per_task_mean,
    }


def _write_svg(rows: List[Dict]) -> None:
    width = 760
    height = 320
    left = 80
    top = 25
    plot_w = 640
    plot_h = 220
    values = [row["average_score"] for row in rows]
    min_v = min(values) - 0.03
    max_v = max(values) + 0.03
    span = max(max_v - min_v, 1e-6)

    points = []
    for i, row in enumerate(rows):
        x = left + int((i / max(len(rows) - 1, 1)) * plot_w)
        y = top + int(((max_v - row["average_score"]) / span) * plot_h)
        points.append((x, y))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="80" y="16" font-size="14" font-family="Arial">Reward Improvement: Baseline vs Mid vs Trained</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    point_text = " ".join(f"{x},{y}" for x, y in points)
    svg.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{point_text}"/>')
    svg.append(f'<text x="{left + 6}" y="{top + 12}" font-size="11" font-family="Arial">y: average reward/grade</text>')
    svg.append(
        f'<text x="{left + plot_w - 175}" y="{top + plot_h + 20}" font-size="11" font-family="Arial">x-axis: policy stage</text>'
    )
    for (x, y), row in zip(points, rows):
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#2563eb"/>')
        svg.append(f'<text x="{x - 18}" y="{top + plot_h + 18}" font-size="11" font-family="Arial">{row["stage"]}</text>')
        svg.append(f'<text x="{x - 20}" y="{y - 8}" font-size="11" font-family="Arial">{row["average_score"]:.3f}</text>')
    svg.append("</svg>")
    SVG_PATH.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = _load_policy_snapshots()
    baseline_policy = snapshots["baseline"]
    mid_policy = snapshots["mid"]
    trained_policy = snapshots["trained"]

    rows = [
        _evaluate_policy(baseline_policy, "baseline", episodes_per_task=12, base_seed=6000),
        _evaluate_policy(mid_policy, "mid", episodes_per_task=12, base_seed=7000),
        _evaluate_policy(trained_policy, "trained", episodes_per_task=12, base_seed=8000),
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "average_score", "std_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row["stage"],
                    "average_score": f"{row['average_score']:.6f}",
                    "std_score": f"{row['std_score']:.6f}",
                }
            )

    JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_svg(rows)

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["evaluation_rows"] = rows
    metrics["reward_progression_artifacts"] = {
        "csv": str(CSV_PATH),
        "json": str(JSON_PATH),
        "svg": str(SVG_PATH),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
