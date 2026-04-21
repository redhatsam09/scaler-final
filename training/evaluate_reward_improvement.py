import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.environment import DataCleaningEnv
from src.graders import MissingValuesGrader, DuplicateHandlingGrader, ComplexValidationGrader
from src.models import Action


OUTPUT_DIR = Path("artifacts")
CSV_PATH = OUTPUT_DIR / "reward_progression.csv"
JSON_PATH = OUTPUT_DIR / "reward_progression.json"
SVG_PATH = OUTPUT_DIR / "reward_progression.svg"


TASKS = [
    ("task_missing_values", MissingValuesGrader),
    ("task_duplicate_handling", DuplicateHandlingGrader),
    ("task_complex_validation", ComplexValidationGrader),
]


def _baseline_actions(columns: list[str]) -> list[Action]:
    return [
        Action(action_type="analyze", target_columns=columns[:2], parameters={}, reasoning="baseline analyze"),
        Action(action_type="impute", target_columns=columns[:2], parameters={"method": "mean"}, reasoning="baseline impute"),
        Action(action_type="deduplicate", target_columns=columns[:2], parameters={}, reasoning="baseline dedup"),
    ]


def _mid_actions(columns: list[str]) -> list[Action]:
    return [
        Action(action_type="analyze", target_columns=columns[:3], parameters={}, reasoning="mid analyze"),
        Action(action_type="impute", target_columns=columns, parameters={"method": "forward_fill"}, reasoning="mid impute"),
        Action(action_type="deduplicate", target_columns=columns[:2], parameters={}, reasoning="mid dedup"),
        Action(action_type="validate", target_columns=columns[:2], parameters={}, reasoning="mid validate"),
    ]


def _trained_actions(task_id: str, columns: list[str]) -> list[Action]:
    if task_id == "task_missing_values":
        return [
            Action(action_type="analyze", target_columns=columns, parameters={}, reasoning="trained analyze"),
            Action(action_type="impute", target_columns=columns, parameters={"method": "forward_fill"}, reasoning="trained impute all"),
            Action(action_type="validate", target_columns=columns, parameters={}, reasoning="trained validate"),
            Action(
                action_type="report_findings",
                target_columns=columns[:1],
                parameters={"include_summary": True, "include_quality_score": True, "include_recommendations": True},
                reasoning="trained report",
            ),
        ]
    if task_id == "task_duplicate_handling":
        subset = [columns[0]] if columns else None
        return [
            Action(action_type="analyze", target_columns=columns, parameters={}, reasoning="trained analyze"),
            Action(action_type="deduplicate", target_columns=columns[:2], parameters={"subset": subset, "keep": "first"}, reasoning="trained dedup"),
            Action(action_type="validate", target_columns=columns[:2], parameters={}, reasoning="trained validate"),
            Action(
                action_type="report_findings",
                target_columns=columns[:1],
                parameters={"include_summary": True, "include_quality_score": True, "include_recommendations": True},
                reasoning="trained report",
            ),
        ]
    return [
        Action(action_type="analyze", target_columns=columns, parameters={}, reasoning="trained analyze"),
        Action(action_type="impute", target_columns=columns, parameters={"method": "forward_fill"}, reasoning="trained impute"),
        Action(action_type="deduplicate", target_columns=columns[:2], parameters={}, reasoning="trained dedup"),
        Action(action_type="validate", target_columns=columns, parameters={}, reasoning="trained validate"),
        Action(
            action_type="report_findings",
            target_columns=columns[:1],
            parameters={"include_summary": True, "include_quality_score": True, "include_recommendations": True},
            reasoning="trained report",
        ),
    ]


def _evaluate_policy(stage: str, seed_offset: int) -> dict:
    env = DataCleaningEnv(seed=2026 + seed_offset)
    task_scores: dict[str, float] = {}

    for idx, (task_id, grader_class) in enumerate(TASKS):
        observation = env.reset(task_id=task_id, seed=2026 + seed_offset + idx)
        columns = observation.column_names

        if stage == "baseline":
            actions = _baseline_actions(columns)
        elif stage == "mid":
            actions = _mid_actions(columns)
        else:
            actions = _trained_actions(task_id, columns)

        for action in actions:
            env.step(action)

        score = float(grader_class.grade(env.current_episode))
        task_scores[task_id] = score

    average_score = sum(task_scores.values()) / len(task_scores)
    return {
        "stage": stage,
        "average_score": average_score,
        "task_scores": task_scores,
    }


def _write_svg(rows: list[dict]) -> None:
    width = 600
    height = 280
    left = 60
    top = 20
    plot_w = 500
    plot_h = 200
    values = [row["average_score"] for row in rows]
    min_v = min(values) - 0.02
    max_v = max(values) + 0.02
    span = max(max_v - min_v, 1e-6)

    points = []
    labels = []
    for i, row in enumerate(rows):
        x = left + int((i / (len(rows) - 1)) * plot_w)
        y = top + int(((max_v - row["average_score"]) / span) * plot_h)
        points.append(f"{x},{y}")
        labels.append((x, row["stage"]))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="16" font-size="14" font-family="Arial">Reward Improvement Over Training Stages</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{" ".join(points)}"/>',
    ]

    for i, row in enumerate(rows):
        x_str, y_str = points[i].split(",")
        x = int(x_str)
        y = int(y_str)
        stage = row["stage"]
        score = row["average_score"]
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#2563eb"/>')
        svg.append(f'<text x="{x - 26}" y="{top + plot_h + 18}" font-size="11" font-family="Arial">{stage}</text>')
        svg.append(f'<text x="{x - 18}" y="{y - 8}" font-size="11" font-family="Arial">{score:.3f}</text>')

    svg.append("</svg>")
    SVG_PATH.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        _evaluate_policy("baseline", 0),
        _evaluate_policy("mid", 10),
        _evaluate_policy("trained", 20),
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "average_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"stage": row["stage"], "average_score": f"{row['average_score']:.6f}"})

    JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_svg(rows)


if __name__ == "__main__":
    main()
