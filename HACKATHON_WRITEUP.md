# OpenEnv Hackathon Writeup

## Problem and Capability Gap

General LLM tool-use agents still struggle with enterprise workflows where objectives conflict, policies drift mid-task, and local shortcuts can hurt global KPIs. We target this gap with a world-modeling environment that forces multi-step decision making under partial observability.

## Environment Summary

This environment simulates CRM, Billing, and Support coordination with:

- Hidden actor trust and incentive conflicts
- Stochastic delegation outcomes and pushback handling
- Mid-episode policy/schema drift (v1 -> v2 -> v3)
- Deceptive recommendations requiring oversight checks
- Explicit economic budgets and action costs
- Natural-language observations and urgency signals

## Agent Interface

Actions available to the agent:

- analyze
- impute
- deduplicate
- validate
- report_findings
- delegate
- resolve_alert
- reconcile_apps
- oversight_review
- inspect_actor
- audit_records
- request_policy_clarification

The environment follows OpenEnv-compatible reset/step/state patterns and is exposed through FastAPI.

## Reward and Anti-Gaming Strategy

Reward is a weighted combination of:

- Action-specific utility
- Progress signal on missing values and duplicates
- Process bonuses for robust sequencing
- Economic efficiency

Penalties include:

- Invalid action penalties
- Loop penalties
- Budget overflow penalties
- Adaptive stale-strategy penalties after policy drift
- Reasoning-quality penalties for underspecified decisions

Anti-hacking guardrails:

- Clarification reward is paid only once per policy version
- Drift penalties continue until drift-aware actions are taken
- Reporting reward requires actual data-quality improvement

## Training Pipeline

- GRPO script: training/grpo_training.py
- Environment-grounded policy search: training/trl_sft_training.py
- Reward progression and ablations: training/evaluate_reward_improvement.py

Local validation and artifact refresh:

```bash
./scripts/run_local_checks.sh
```

## Results

From artifacts/reward_progression.json:

- Baseline: 0.488
- Mid: 0.677
- Trained: 0.701
- Improvement: +0.214 (+43.8%)

From artifacts/ablation_no_actor_actions.json:

- Full policy (enterprise task): 0.808
- Without actor-facing actions: 0.424
- Delta: +0.384

From artifacts/heldout_drift_scenario.json:

- Held-out hard drift score: 0.831

## Reproducibility and Deployment

- API server: server/app.py
- OpenEnv manifest: openenv.yaml
- Dockerized Space deployment supported
- Token-only deployment script: scripts/deploy_hf_space.sh

Deploy command:

```bash
export HF_TOKEN="<your_hf_write_token>"
export HF_SPACE_ID="samdutta123/scaler-final-openenv"
./scripts/deploy_hf_space.sh
```
