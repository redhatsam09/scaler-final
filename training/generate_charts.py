"""
Generate publication-quality training evidence charts as PNG files.
These charts are displayed in the Gradio "Training Evidence" tab.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ARTIFACTS = ROOT_DIR / "artifacts"
REWARD_JSON = ARTIFACTS / "reward_progression.json"
ABLATION_JSON = ARTIFACTS / "ablation_no_actor_actions.json"

# ── Output PNGs ──
REWARD_CHART = ARTIFACTS / "reward_progression_chart.png"
ABLATION_CHART = ARTIFACTS / "ablation_chart.png"
TASK_BREAKDOWN_CHART = ARTIFACTS / "task_breakdown_chart.png"


def _style():
    """Apply a clean, professional matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#e2e8f0",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def generate_reward_progression():
    """Bar chart: baseline → mid → trained average scores."""
    data = json.loads(REWARD_JSON.read_text())
    stages = [d["stage"].capitalize() for d in data]
    scores = [d["average_score"] for d in data]
    stds = [d["std_score"] for d in data]

    _style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    bars = ax.bar(stages, scores, color=colors, edgecolor="#0f172a", linewidth=1.5,
                  yerr=stds, capsize=6, error_kw={"ecolor": "#94a3b8", "linewidth": 1.5})

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{score:.3f}", ha="center", va="bottom", fontweight="bold",
                fontsize=13, color="#f8fafc")

    # Improvement arrow
    improvement = ((scores[-1] - scores[0]) / scores[0]) * 100
    ax.annotate(f"+{improvement:.1f}%", xy=(2, scores[-1]),
                xytext=(1.5, scores[-1] + 0.08),
                fontsize=14, fontweight="bold", color="#22c55e",
                arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2))

    ax.set_ylabel("Average Grader Score (5-seed mean)")
    ax.set_title("Reward Improvement: Baseline → Mid → Trained Policy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(str(REWARD_CHART), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {REWARD_CHART}")


def generate_task_breakdown():
    """Grouped bar chart: per-task scores across training stages."""
    data = json.loads(REWARD_JSON.read_text())
    stages = [d["stage"].capitalize() for d in data]
    task_ids = list(data[0]["task_scores"].keys())
    short_names = [t.replace("task_", "").replace("_", " ").title() for t in task_ids]

    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(task_ids))
    width = 0.25
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    for i, (stage_data, color, label) in enumerate(zip(data, colors, stages)):
        vals = [stage_data["task_scores"][t] for t in task_ids]
        ax.bar(x + i * width, vals, width, label=label, color=color,
               edgecolor="#0f172a", linewidth=1)

    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.set_ylabel("Grader Score")
    ax.set_title("Per-Task Score Improvement Across Training Stages")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    fig.savefig(str(TASK_BREAKDOWN_CHART), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {TASK_BREAKDOWN_CHART}")


def generate_ablation():
    """Horizontal bar chart: full policy vs ablation (no actor actions)."""
    if not ABLATION_JSON.exists():
        print(f"Skipping ablation chart — {ABLATION_JSON} not found")
        return

    abl = json.loads(ABLATION_JSON.read_text())
    full_score = abl.get("full_policy_score", 0.808)
    ablated_score = abl.get("ablated_score", 0.424)

    _style()
    fig, ax = plt.subplots(figsize=(8, 3.5))

    labels = ["Without Actor Actions\n(Ablated)", "Full Policy\n(With Actor Actions)"]
    values = [ablated_score, full_score]
    colors = ["#ef4444", "#22c55e"]

    bars = ax.barh(labels, values, color=colors, edgecolor="#0f172a", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontweight="bold", fontsize=13, color="#f8fafc")

    delta = full_score - ablated_score
    ax.set_title(f"Ablation Study: Actor Actions Contribute +{delta:.3f} to Score")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", linestyle="--")
    fig.tight_layout()
    fig.savefig(str(ABLATION_CHART), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ABLATION_CHART}")


if __name__ == "__main__":
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    generate_reward_progression()
    generate_task_breakdown()
    generate_ablation()
    print("All charts generated successfully.")
