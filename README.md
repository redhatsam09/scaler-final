---
title: scaler-final-submission
emoji: "🐳"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Multi-App Enterprise Orchestration Environment

An OpenEnv environment for **Theme #3 World Modeling (Professional Tasks)** with strong overlap to:
- **Theme #1 Multi-Agent Interactions** (delegation + actor escalations),
- **Theme #2 Long-Horizon Planning** (schema drift + delayed coordination effects),
- Patronus-style **schema/policy drift** behavior.

The environment simulates CRM + Billing + Support workflows in one partially observable world. Agents must choose tool actions, negotiate conflicting actor incentives, handle contract drift, detect deceptive recommendations, and improve enterprise KPIs over multi-step episodes.

![Hidden state to action to KPI to grade loop](artifacts/world_model_flow.svg)

## Why this is competitive

This project is designed around the judging rubric:
- **Innovation (40%)**: multi-app orchestration, conflicting actor incentives, deceptive actor oversight, dynamic policy drift, economic cost tradeoffs, and curriculum difficulty.
- **Storytelling (30%)**: clear demo script plus generated figures for hidden state flow and failure-before/success-after trajectories.
- **Reward improvement evidence (20%)**: 5-seed mean/std, held-out hard-drift scenario, and no-actor-action ablation from real environment rollouts.
- **Reward/training pipeline (10%)**: environment-grounded policy optimization plus explicit TRL/Unsloth notebook artifact.

## Environment design

### Tasks

- `task_missing_values`: CRM quality repair.
- `task_duplicate_handling`: Billing deduplication and consistency.
- `task_complex_validation`: Support quality validation under mixed constraints.
- `task_enterprise_orchestration`: **new hard task** combining CRM/Billing/Support with multi-actor delegation and schema drift.

### Agent actions

- `analyze`
- `impute`
- `deduplicate`
- `validate`
- `report_findings`
- `delegate`
- `resolve_alert`
- `reconcile_apps`
- `oversight_review`

### Actor incentives and conflicts

The enterprise task is not only workflow automation. Each episode exposes actor objectives and conflicts:

- `finance_bot`: minimizes write-offs and operation cost.
- `support_lead`: protects SLA and critical ticket backlog.
- `sales_ops`: maximizes conversion and account coverage.
- `compliance_officer`: enforces the latest policy/T&C version.
- `analytics_assistant`: explains KPIs, but can occasionally issue a deceptive shortcut recommendation.

Key conflicts:

- Finance rejects expensive fixes while support asks for costly escalations.
- Sales prioritizes high-conversion accounts while support prioritizes SLA risk.
- Analytics may propose KPI shortcuts while compliance requires explainable policy-safe changes.

### World modeling dynamics

- Hidden mutable dataset state.
- Partial observations (`missing_values`, schema/dtypes, KPI snapshots, actor messages).
- Mid-episode schema/policy drift in enterprise task:
  - new `compliance_tier` field,
  - invoice status contract change (`pending` → `awaiting_payment`),
  - new compliance validation requirement.
- Dynamic T&C updates during the same episode:
  - policy v2 activates compliance-tier checks,
  - policy v3 requires high-risk EU accounts to satisfy stricter ticket/invoice closure rules,
  - stale strategies after policy changes are penalized.
- Curriculum difficulty:
  - `easy`, `medium`, and `hard` change drift timing, deception probability, cost budget, and cost noise.
- Economic cost model:
  - each action has an operation cost,
  - reward trades off quality, SLA, conversion, compliance, latency, and remaining budget.
- Deceptive actor + oversight:
  - `analytics_assistant` can recommend a wrong shortcut,
  - `oversight_review` detects and explains the deceptive recommendation.

### Anti-gaming reward design

- Per-step shaped progress signal.
- Rubric-style task graders.
- Loop penalties for repeated action spam.
- Over-deletion penalties for destructive shortcuts.
- KPI-aware scoring (quality/compliance/latency).
- Actor-alignment scoring (finance cost efficiency, support SLA health, sales conversion health).
- Budget overflow and stale-policy penalties.

## API

Endpoints:
- `POST /reset`
- `POST /step`
- `POST /state`
- `POST /grade`
- `GET /health`

`/reset` returns `session_id`; reuse it across step/state/grade calls.

## Setup

```bash
git clone https://github.com/redhatsam09/scaler-final.git
cd scaler-final
pip install -r requirements.txt
pip install -e .
```

## Run

Start server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run deterministic inference:

```bash
INFERENCE_BACKEND=local INFERENCE_SEED=2026 python inference.py
```

Run world-modeling demo:

```bash
python world_modeling_demo.py
```

## Training + Evidence pipeline (environment-grounded)

1) Train policy from environment rollouts:

```bash
python training/trl_sft_training.py
```

Generates:
- `artifacts/learned_policy.json`
- `artifacts/training_curve.csv`
- `artifacts/training_curve.json`
- `artifacts/training_curve.svg`
- `artifacts/trl_sft_training_metrics.json`

2) Evaluate baseline vs mid vs trained:

```bash
python training/evaluate_reward_improvement.py
```

Generates:
- `artifacts/reward_progression.csv`
- `artifacts/reward_progression.json`
- `artifacts/reward_progression.svg`
- `artifacts/ablation_no_actor_actions.json`
- `artifacts/heldout_drift_scenario.json`
- `artifacts/world_model_flow.svg`
- `artifacts/failure_success_trajectory.svg`

![Failure before and success after trajectory](artifacts/failure_success_trajectory.svg)

3) Optional explicit TRL/Unsloth compliance notebook:

```text
training/trl_unsloth_compliance_notebook.ipynb
```

This notebook is kept as a judge-friendly TRL/Unsloth artifact. The stronger training evidence remains the environment-grounded rollout pipeline above.

## Validation commands

```bash
openenv validate
python world_modeling_demo.py
python inference.py
python training/trl_sft_training.py
python training/evaluate_reward_improvement.py
```

## Submission links

- Hugging Face Space URL: `https://huggingface.co/spaces/samdutta123/scaler-final-openenv`
- Live API base URL: `https://samdutta123-scaler-final-openenv.hf.space`
- Colab notebook URL: `REPLACE_WITH_FINAL_COLAB_URL`
- Mini-blog URL: `REPLACE_WITH_FINAL_BLOG_URL`
- Mini-video URL (<2 min): `REPLACE_WITH_FINAL_VIDEO_URL`

> Replace placeholders before final submission.
