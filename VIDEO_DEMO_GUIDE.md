# Video Demo Guide

Use this guide to record a short walkthrough that shows what the project does, why it fits Theme #3 World Modeling, and how someone can validate it.

Recommended length: 90-120 seconds.

## Core Message

This project is an OpenEnv-compatible enterprise workflow environment. The hidden world state is a mutable business dataset. The agent only sees partial observations such as schema, shape, missing-value counts, duplicate counts, and progress. It must take multi-step actions that change the world state and improve a grader score.

The environment covers three professional enterprise tasks:
- CRM contact cleanup: `task_missing_values`
- Billing invoice duplicate handling: `task_duplicate_handling`
- Support ticket validation: `task_complex_validation`

## Before Recording

Open two terminal panes if possible.

Pane 1: run validation/demo commands.

Pane 2: optionally run the API server and API curl calls.

Start in the repository root:

```bash
cd /workspaces/scaler-final
git status --short --branch
```

Expected branch state should show `main...origin/main`. The untracked `.codex` file can be ignored.

## 2 Minute Storyboard

0:00-0:15: Problem and theme fit

Say:

> This is a Theme #3 World Modeling project for professional enterprise workflows. It simulates data operations where an agent must maintain state, observe partial information, and take actions that causally change a hidden business dataset.

Show:

```bash
sed -n '1,80p' README.md
```

0:15-0:45: Show the world model behavior

Run:

```bash
python world_modeling_demo.py
```

Point out these lines:

```text
Hidden world state: enterprise business dataset
Visible observation: schema, missing-value counts, shape, progress summary
Agent actions: analyze -> clean -> validate -> report
```

Then explain one task:

```text
reset: dataset=crm_contacts shape=(132, 7) missing=108 duplicates=24 actions=0
step=2 action=impute ... missing 108->2
step=3 action=deduplicate ... duplicates 22->0
final: ... score=0.990
```

Say:

> This proves the environment is dynamic. The agent does not just answer a static question. It changes the underlying dataset, and the grader measures whether the change improved the enterprise workflow.

0:45-1:10: Show standard validation

Run:

```bash
openenv validate
python inference.py
```

Expected key outputs:

```text
[OK] scaler-final: Ready for multi-mode deployment
[SUMMARY] average_score=0.990000
```

Say:

> The inference runner can use Gemini through the OpenAI-compatible Gemini API, but it also has a deterministic local fallback so the demo remains reproducible if a temporary API key is rate limited.

1:10-1:40: Show API interaction

Run against the live Hugging Face Space:

```bash
curl -sS https://samdutta123-scaler-final-openenv.hf.space/health
```

Expected:

```json
{"status":"ok"}
```

Start an episode:

```bash
curl -sS -X POST 'https://samdutta123-scaler-final-openenv.hf.space/reset' \
  -H 'Content-Type: application/json' \
  -d '{"task_id":"task_missing_values","seed":2026}'
```

Copy the returned `session_id`, then step:

```bash
curl -sS -X POST 'https://samdutta123-scaler-final-openenv.hf.space/step' \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id":"PASTE_SESSION_ID",
    "action":{
      "action_type":"analyze",
      "target_columns":["email","phone"],
      "parameters":{},
      "reasoning":"Inspect missing contact fields first"
    }
  }'
```

Check state:

```bash
curl -sS -X POST 'https://samdutta123-scaler-final-openenv.hf.space/state?session_id=PASTE_SESSION_ID'
```

Check score:

```bash
curl -sS -X POST 'https://samdutta123-scaler-final-openenv.hf.space/grade?task_id=task_missing_values&session_id=PASTE_SESSION_ID'
```

Say:

> The API is session-based. Reset creates a world instance, step mutates that instance, state shows the new state, and grade scores the current episode.

1:40-2:00: Usefulness and close

Say:

> This is useful because it trains and evaluates agents on realistic enterprise workflow behavior: tracking state, choosing tools, handling partial observations, and improving measurable business data quality over multiple steps.

Show evidence artifacts:

```bash
cat artifacts/reward_progression.csv
```

Expected:

```text
stage,average_score
baseline,0.655990
mid,0.958862
trained,0.990000
```

## Gemini Configuration

Gemini model currently configured:

```bash
MODEL_NAME=gemini-3-flash-preview
INFERENCE_BACKEND=gemini
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Google's Gemini model docs list `gemini-3-flash-preview` as the Gemini 3 Flash Preview model code:

https://ai.google.dev/gemini-api/docs/models/gemini

Run Gemini mode locally:

```bash
GEMINI_API_KEY=YOUR_KEY INFERENCE_BACKEND=gemini MODEL_NAME=gemini-3-flash-preview python inference.py
```

If the key is rate limited, the runner logs `MODEL_ERROR` and falls back to the deterministic local policy for that task. That fallback is intentional for reliable demos.

## Commands To Keep Handy

```bash
cd /workspaces/scaler-final
openenv validate
python world_modeling_demo.py
python inference.py
python training/evaluate_reward_improvement.py
cat artifacts/reward_progression.csv
```

Live Space:

```text
https://huggingface.co/spaces/samdutta123/scaler-final-openenv
https://samdutta123-scaler-final-openenv.hf.space
```

## One-Sentence Pitch

This environment teaches and evaluates agents to maintain a world model over a mutable enterprise dataset, choose workflow actions under partial observability, and improve measurable data quality through multi-step tool interaction.
