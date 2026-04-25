# 🏢 OpenEnv Hackathon Writeup
## Enterprise Orchestration RL Environment — Theme #3.1 World Modeling

---

## Problem and Capability Gap

General LLM tool-use agents still struggle with enterprise workflows where objectives conflict, policies drift mid-task, and local shortcuts can hurt global KPIs.

**The gap:** Current agents can call tools in a predefined script but cannot:
- Model hidden actor incentives and detect deception
- Adapt strategy when compliance policies shift mid-episode
- Balance competing KPIs under an economic budget

We close this gap with a **world-modeling environment** that forces multi-step decision making under partial observability.

---

## Environment Summary

This environment simulates CRM, Billing, and Support coordination across 4 tasks of increasing complexity. The flagship task (`task_enterprise_orchestration`) includes:

| Feature | Description |
|---------|-------------|
| 🎭 **5 Actors with Hidden Incentives** | finance_bot, support_lead, sales_ops, compliance_officer, analytics_assistant |
| 🕵️ **Deceptive Recommendations** | analytics_assistant may suggest KPI shortcuts detectable only via `oversight_review` |
| 🔄 **Schema Drift (v1→v2→v3)** | Mid-episode field renames, new compliance tiers, evolving T&C |
| 💰 **Economic Budget** | Each action costs budget; overflow is penalized |
| 🔒 **Stochastic Delegation** | Actor pushback probability depends on hidden trust score |
| 👁️ **Partial Observability** | Agent only sees surface-level KPI snapshots, not hidden state |

---

## Agent Interface

The agent has access to **12 actions**:
`analyze`, `impute`, `deduplicate`, `validate`, `report_findings`, `delegate`, `resolve_alert`, `reconcile_apps`, `oversight_review`, `inspect_actor`, `audit_records`, `request_policy_clarification`

Optimal policy for the enterprise task requires:
1. **Inspect-before-delegate** — check actor trust before assigning work
2. **Oversight-before-report** — detect and flag deceptive advice
3. **Clarify-after-drift** — request T&C updates when schema drift is detected
4. **Reconcile-cross-app** — resolve CRM↔Billing↔Support mismatches

---

## Reward and Anti-Gaming Strategy

Reward is a multi-component signal:

**Positive:**
- Action-specific utility (imputation reduces missing values, delegation succeeds)
- Process bonuses for smart sequencing (analyze-first, inspect-before-delegate)
- Economic efficiency bonuses

**Negative (anti-gaming):**
- Loop penalties for repeating the same action
- Budget overflow penalties
- Stale-strategy penalties after policy drift (persist until drift-aware actions executed)
- Reasoning-quality penalties for underspecified actions
- Clarification reward paid only **once per policy version** (no spam farming)
- Report reward requires actual data improvement (no free points for empty reports)

---

## Training Pipeline

Our RL training uses **GRPO (Group Relative Policy Optimization)** via TRL + Unsloth:

1. **Environment generates state observations** → Natural language prompts fed to the model
2. **Model generates 4 candidate actions** per prompt
3. **Each action runs through `env.step(action)`** — verifiable reward from the environment
4. **Reward = 0.6 × step_reward + 0.4 × episode_grade + 0.1 × format_bonus**
5. **GRPO updates the policy** to prefer actions yielding higher expected reward

| Training File | Purpose |
|--------------|---------|
| `training/grpo_training.py` | Main GRPO training loop |
| `training/trl_unsloth_compliance_notebook.ipynb` | Colab-ready notebook with Unsloth |
| `training/trl_sft_training.py` | Policy search baseline |
| `training/evaluate_reward_improvement.py` | Evaluation + artifact generation |

```bash
# Local validation and artifact refresh
./scripts/run_local_checks.sh
```

---

## Results

### Reward Progression (5-seed mean)

| Stage | Score | Δ vs Baseline |
|-------|-------|---------------|
| Baseline (random policy) | 0.488 | — |
| Mid-training | 0.677 | +0.189 (+38.7%) |
| **Trained** | **0.701** | **+0.214 (+43.8%)** |

### Ablation Study — Actor-Facing Actions Matter

| Policy | Enterprise Score |
|--------|-----------------|
| Full policy (all 12 actions) | 0.808 |
| Ablated (no actor actions) | 0.424 |
| **Δ from actor actions** | **+0.384** |

### Generalization — Held-out Hard Drift Scenario

> Score: **0.831** on unseen episodes with hard-mode drift, deception, and tighter budget.

---

## Reproducibility and Deployment

- API server: `server/app.py`
- OpenEnv manifest: `openenv.yaml`
- Dockerized Space deployment supported
- Token-only deployment script: `scripts/deploy_hf_space.sh`

```bash
export HF_TOKEN="<your_hf_write_token>"
export HF_SPACE_ID="samdutta123/scaler-final-openenv"
./scripts/deploy_hf_space.sh
```
