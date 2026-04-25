## 🧠 World Modeling — How Our Environment Tests It

World modeling is the ability of an agent to maintain an internal representation of hidden state and update it from partial observations. Our environment rigorously tests this in four ways:

### 1. Hidden Actor Trust (Theory of Mind)
Actors like `finance_bot` have a hidden trust score (0.0–1.0). The agent only sees surface-level actor messages. To make smart delegation decisions, the agent must use `inspect_actor` to reveal trust levels before committing budget to delegation. Agents that blindly delegate suffer high pushback rates.

### 2. Deceptive Recommendations (Belief Revision)
The `analytics_assistant` can recommend shortcuts like "mark all overdue invoices as paid" which appear helpful but violate compliance. The agent must detect this via `oversight_review`. Agents that accept advice without verification receive a deception penalty.

### 3. Schema Drift (Dynamic State Tracking)
At random steps, the environment applies schema drift: field names change, new required columns appear, compliance tiers shift. The agent must request `request_policy_clarification` and then execute drift-aware actions (validate, reconcile_apps) to avoid stale-strategy penalties.

### 4. Cross-App Inconsistency (Spatial World Model)
CRM, Billing, and Support data can diverge — the same account_id has different `compliance_tier` values across systems. The agent must build a model of cross-system inconsistencies and use `reconcile_apps` to resolve conflicts before reporting.

### Why This Matters for LLM Training
Current LLMs struggle with these tasks because they require:
- Tracking hidden variables across multiple steps
- Updating beliefs based on new evidence
- Resisting deceptive shortcuts
- Adapting strategy when the world changes mid-episode

Training on our environment should improve these capabilities measurably.
