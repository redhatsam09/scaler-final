import csv
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

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
from src.policy_space import ACTION_SPACE, ACTOR_ACTIONS, DEFAULT_REPORT_PARAMS, EXTENDED_REPORT_PARAMS


OUTPUT_DIR = Path("artifacts")
CSV_PATH = OUTPUT_DIR / "reward_progression.csv"
JSON_PATH = OUTPUT_DIR / "reward_progression.json"
SVG_PATH = OUTPUT_DIR / "reward_progression.svg"
ABLATION_PATH = OUTPUT_DIR / "ablation_no_actor_actions.json"
HELDOUT_DRIFT_PATH = OUTPUT_DIR / "heldout_drift_scenario.json"
FLOW_FIGURE_PATH = OUTPUT_DIR / "world_model_flow.svg"
TRAJECTORY_FIGURE_PATH = OUTPUT_DIR / "failure_success_trajectory.svg"
POLICY_PATH = OUTPUT_DIR / "learned_policy.json"
POLICY_SNAPSHOTS_PATH = OUTPUT_DIR / "policy_snapshots.json"
METRICS_PATH = OUTPUT_DIR / "trl_sft_training_metrics.json"
SEED_GROUPS = (6100, 6200, 6300, 6400, 6500)


TASK_SPECS: Sequence[tuple[str, Callable]] = (
    ("task_missing_values", MissingValuesGrader),
    ("task_duplicate_handling", DuplicateHandlingGrader),
    ("task_complex_validation", ComplexValidationGrader),
    ("task_enterprise_orchestration", EnterpriseOrchestrationGrader),
)

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
            "parameters": dict(DEFAULT_REPORT_PARAMS),
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
    difficulty: str | None = None,
    task_filter: Sequence[str] | None = None,
) -> Dict:
    env = DataCleaningEnv(seed=base_seed)
    selected_specs = [(task_id, grader) for task_id, grader in TASK_SPECS if task_filter is None or task_id in task_filter]
    task_scores: Dict[str, List[float]] = {task_id: [] for task_id, _ in selected_specs}
    episode_records: List[Dict] = []
    for task_idx, (task_id, grader) in enumerate(selected_specs):
        for episode_idx in range(episodes_per_task):
            seed = base_seed + task_idx * 100 + episode_idx
            observation = env.reset(task_id=task_id, seed=seed, difficulty=difficulty)
            for proto in policy.get(task_id, []):
                action = _action_from_proto(task_id, proto, observation.column_names)
                observation, _, done, _ = env.step(action)
                if done:
                    break
            score = float(grader.grade(env.current_episode))
            task_scores[task_id].append(score)
            state = env.state()
            episode_records.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "difficulty": state.get("difficulty"),
                    "score": round(score, 6),
                    "policy_version": state.get("policy_version"),
                    "drift_active": state.get("drift_active"),
                    "deception_detected": state.get("deception_detected"),
                    "economic_status": state.get("economic_status", {}),
                    "kpi_snapshot": state.get("kpi_snapshot", {}),
                }
            )

    per_task_mean = {task_id: float(np.mean(values)) for task_id, values in task_scores.items()}
    all_scores = [score for scores in task_scores.values() for score in scores]
    return {
        "stage": stage,
        "average_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores)),
        "task_scores": per_task_mean,
        "episodes": episode_records,
    }


def _evaluate_across_seed_groups(policy: Dict[str, List[Dict]], stage: str, episodes_per_task: int) -> Dict:
    seed_rows = [
        _evaluate_policy(policy, stage, episodes_per_task=episodes_per_task, base_seed=seed)
        for seed in SEED_GROUPS
    ]
    stage_scores = [row["average_score"] for row in seed_rows]
    task_ids = [task_id for task_id, _ in TASK_SPECS]
    return {
        "stage": stage,
        "average_score": float(np.mean(stage_scores)),
        "std_score": float(np.std(stage_scores)),
        "seed_group_scores": [
            {
                "base_seed": seed,
                "average_score": round(row["average_score"], 6),
                "std_score": round(row["std_score"], 6),
            }
            for seed, row in zip(SEED_GROUPS, seed_rows)
        ],
        "task_scores": {
            task_id: float(np.mean([row["task_scores"][task_id] for row in seed_rows]))
            for task_id in task_ids
        },
    }


def _remove_actor_actions(policy: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    ablated: Dict[str, List[Dict]] = {}
    for task_id, actions in policy.items():
        replacement = {"action_type": "validate", "parameters": {}}
        if task_id == "task_enterprise_orchestration":
            replacement = {"action_type": "reconcile_apps", "parameters": {"join_key": "account_id"}}
        filtered = [dict(action) for action in actions if action.get("action_type") not in ACTOR_ACTIONS]
        while len(filtered) < len(actions):
            filtered.insert(max(len(filtered) - 1, 0), dict(replacement))
        if filtered:
            filtered[-1] = {
                "action_type": "report_findings",
                "parameters": dict(EXTENDED_REPORT_PARAMS),
            }
        ablated[task_id] = filtered
    return ablated


def _write_flow_figure() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="920" height="260" viewBox="0 0 920 260">
<rect width="920" height="260" fill="#ffffff"/>
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#334155"/>
  </marker>
</defs>
<text x="28" y="34" font-family="Arial" font-size="18" font-weight="700" fill="#111827">World Modeling Loop</text>
<g font-family="Arial">
  <rect x="30" y="70" width="170" height="92" rx="6" fill="#e0f2fe" stroke="#0369a1"/>
  <text x="54" y="103" font-size="14" font-weight="700" fill="#0f172a">Hidden State</text>
  <text x="54" y="126" font-size="12" fill="#334155">CRM + billing + support</text>
  <text x="54" y="145" font-size="12" fill="#334155">drift, costs, actors</text>
  <line x1="205" y1="116" x2="286" y2="116" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="290" y="70" width="170" height="92" rx="6" fill="#fef3c7" stroke="#b45309"/>
  <text x="330" y="103" font-size="14" font-weight="700" fill="#0f172a">Agent Actions</text>
  <text x="315" y="126" font-size="12" fill="#334155">delegate, reconcile,</text>
  <text x="315" y="145" font-size="12" fill="#334155">validate, oversight</text>
  <line x1="465" y1="116" x2="546" y2="116" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="550" y="70" width="170" height="92" rx="6" fill="#dcfce7" stroke="#15803d"/>
  <text x="590" y="103" font-size="14" font-weight="700" fill="#0f172a">KPI Changes</text>
  <text x="576" y="126" font-size="12" fill="#334155">quality, SLA, cost,</text>
  <text x="576" y="145" font-size="12" fill="#334155">conversion, compliance</text>
  <line x1="725" y1="116" x2="806" y2="116" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="810" y="70" width="80" height="92" rx="6" fill="#ede9fe" stroke="#6d28d9"/>
  <text x="826" y="108" font-size="14" font-weight="700" fill="#0f172a">Final</text>
  <text x="823" y="130" font-size="14" font-weight="700" fill="#0f172a">Grade</text>
  <path d="M635 174 C590 225, 330 225, 115 174" fill="none" stroke="#64748b" stroke-dasharray="5 5"/>
  <text x="304" y="232" font-size="12" fill="#475569">Partial observations expose only schema, messages, and KPI snapshots.</text>
</g>
</svg>"""
    FLOW_FIGURE_PATH.write_text(svg, encoding="utf-8")


def _write_trajectory_figure(ablation: Dict, heldout: Dict) -> None:
    before = ablation["no_actor_actions"]["average_score"]
    after = ablation["full_policy"]["average_score"]
    heldout_score = heldout["average_score"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='#94a3b8')
    ax.set_axisbelow(True)
    
    categories = ['No Actor Actions', 'Full Policy', 'Held-out Hard Drift']
    scores = [before, after, heldout_score]
    colors = ['#fca5a5', '#86efac', '#93c5fd']
    edge_colors = ['#b91c1c', '#15803d', '#1d4ed8']

    bars = ax.bar(categories, scores, color=colors, edgecolor=edge_colors, linewidth=1.5, width=0.6)
    
    ax.set_ylim(0, max(scores) + 0.15)
    ax.set_ylabel('Mean Grade', fontweight='bold', color='#334155')
    ax.set_title('Failure Before / Success After Trajectory', fontweight='bold', pad=15, fontsize=12, color='#0f172a')
    
    ax.tick_params(colors='#475569')
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#0f172a')

    # Add description text below
    plt.figtext(0.5, 0.02, 
                "Failure: ignores finance/support/sales conflict and deceptive advice.\n"
                "Success: negotiates actors, checks oversight, adapts to T&C drift.", 
                ha="center", fontsize=10, color="#475569")
                
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(str(TRAJECTORY_FIGURE_PATH))
    plt.savefig(str(TRAJECTORY_FIGURE_PATH).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _write_svg(rows: List[Dict]) -> None:
    stages = [row["stage"] for row in rows]
    means = [row["average_score"] for row in rows]
    stds = [row["std_score"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Modern styling
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    ax.grid(True, linestyle='--', alpha=0.4, color='#94a3b8')
    
    ax.errorbar(stages, means, yerr=stds, fmt='o-', color='#3b82f6', 
                linewidth=2.5, markersize=10, capsize=5, capthick=1.5, markerfacecolor='#1d4ed8')
    
    ax.set_title('GRPO RL Reward Optimization Progression', fontweight='bold', pad=15, fontsize=12, color='#0f172a')
    ax.set_xlabel('Training Stage (Epochs)', fontweight='bold', color='#334155')
    ax.set_ylabel('Mean Episode Reward / Grade', fontweight='bold', color='#334155')
    
    ax.tick_params(colors='#475569')
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
    
    # Add value annotations
    for i, (stage, mean) in enumerate(zip(stages, means)):
        ax.annotate(f'{mean:.3f}', 
                    xy=(stage, mean),
                    xytext=(-5, 12),
                    textcoords="offset points",
                    ha='right', va='bottom', fontweight='bold', color='#1d4ed8', fontsize=10)
                    
    plt.tight_layout()
    plt.savefig(str(SVG_PATH))
    plt.savefig(str(SVG_PATH).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = _load_policy_snapshots()
    baseline_policy = snapshots["baseline"]
    mid_policy = snapshots["mid"]
    trained_policy = snapshots["trained"]

    rows = [
        _evaluate_across_seed_groups(baseline_policy, "baseline", episodes_per_task=5),
        _evaluate_across_seed_groups(mid_policy, "mid", episodes_per_task=5),
        _evaluate_across_seed_groups(trained_policy, "trained", episodes_per_task=5),
    ]

    full_policy_enterprise = _evaluate_policy(
        trained_policy,
        "full_policy",
        episodes_per_task=12,
        base_seed=8100,
        task_filter=["task_enterprise_orchestration"],
    )
    no_actor_enterprise = _evaluate_policy(
        _remove_actor_actions(trained_policy),
        "no_actor_actions",
        episodes_per_task=12,
        base_seed=8100,
        task_filter=["task_enterprise_orchestration"],
    )
    ablation = {
        "description": "Enterprise task ablation comparing full policy against the same action program with actor-facing actions removed.",
        "actor_actions_removed": sorted(ACTOR_ACTIONS),
        "full_policy": {
            "average_score": round(full_policy_enterprise["average_score"], 6),
            "std_score": round(full_policy_enterprise["std_score"], 6),
            "task_scores": {k: round(v, 6) for k, v in full_policy_enterprise["task_scores"].items()},
        },
        "no_actor_actions": {
            "average_score": round(no_actor_enterprise["average_score"], 6),
            "std_score": round(no_actor_enterprise["std_score"], 6),
            "task_scores": {k: round(v, 6) for k, v in no_actor_enterprise["task_scores"].items()},
        },
        "delta_full_minus_ablation": round(
            full_policy_enterprise["average_score"] - no_actor_enterprise["average_score"],
            6,
        ),
    }
    ABLATION_PATH.write_text(json.dumps(ablation, indent=2), encoding="utf-8")

    heldout = _evaluate_policy(
        trained_policy,
        "heldout_hard_drift",
        episodes_per_task=16,
        base_seed=9900,
        difficulty="hard",
        task_filter=["task_enterprise_orchestration"],
    )
    heldout_summary = {
        "description": "Held-out enterprise scenario with hard curriculum drift, higher deception probability, tighter budget, and faster policy updates.",
        "average_score": round(heldout["average_score"], 6),
        "std_score": round(heldout["std_score"], 6),
        "task_scores": {k: round(v, 6) for k, v in heldout["task_scores"].items()},
        "episodes": heldout["episodes"],
    }
    HELDOUT_DRIFT_PATH.write_text(json.dumps(heldout_summary, indent=2), encoding="utf-8")
    _write_flow_figure()
    _write_trajectory_figure(ablation, heldout_summary)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "five_seed_mean", "five_seed_std"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row["stage"],
                    "five_seed_mean": f"{row['average_score']:.6f}",
                    "five_seed_std": f"{row['std_score']:.6f}",
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
        "ablation_no_actor_actions": str(ABLATION_PATH),
        "heldout_drift_scenario": str(HELDOUT_DRIFT_PATH),
        "world_model_flow": str(FLOW_FIGURE_PATH),
        "failure_success_trajectory": str(TRAJECTORY_FIGURE_PATH),
    }
    metrics["ablation_no_actor_actions"] = ablation
    metrics["heldout_drift_scenario"] = {
        "average_score": heldout_summary["average_score"],
        "std_score": heldout_summary["std_score"],
        "task_scores": heldout_summary["task_scores"],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
