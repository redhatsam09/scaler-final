# Video Demo Guide (3-minute pitch + 2-minute Q&A prep)

Use this script to align with judging criteria: innovation, storytelling, reward improvement evidence, and training pipeline quality.

## Suggested 3-minute pitch flow

### 0:00-0:30 Problem + novelty

Say:

> We built a multi-app enterprise orchestration environment where an agent manages CRM, Billing, and Support workflows under partial observability. Mid-episode schema and policy contracts drift, so the agent must adapt and coordinate actors.

Show:

```bash
sed -n '1,120p' README.md
```

### 0:30-1:20 Environment behavior and world modeling

Run:

```bash
python world_modeling_demo.py
```

Point out:
- actor messages,
- drift notice activation,
- KPI changes and final grade.

Highlight:

> The agent is not solving a static prompt; it is operating a mutable world state through tool actions.

### 1:20-2:20 Training + measurable improvement

Run:

```bash
python training/trl_sft_training.py
python training/evaluate_reward_improvement.py
cat artifacts/reward_progression.csv
```

Say:

> The policy is trained by interacting with the environment. We show baseline, mid, and trained reward on held-out seeds.

Also show:

```bash
cat artifacts/trl_sft_training_metrics.json
```

### 2:20-3:00 API + reproducibility

Run:

```bash
openenv validate
curl -sS https://samdutta123-scaler-final-openenv.hf.space/health
```

Optional session flow:

```bash
curl -sS -X POST 'https://samdutta123-scaler-final-openenv.hf.space/reset' \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"task_enterprise_orchestration","seed":2026}'
```

Use returned `session_id` for `/step`, `/state`, `/grade`.

## Common Q&A answers

**Q: Why is this innovative?**  
A: It combines multi-agent delegation, schema drift, and cross-app reconciliation in one partially observable workflow.

**Q: How do you prevent reward hacking?**  
A: We use anti-loop and over-deletion penalties, plus rubric-style task graders that require meaningful state improvement.

**Q: What proves learning?**  
A: Environment-grounded training artifacts plus baseline/mid/trained comparisons on held-out seeds.
