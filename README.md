---
title: scaler-final-submission
emoji: "🐳"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Data Cleaning Environment (OpenEnv)

This repository provides an OpenEnv-compatible data cleaning environment with:
1. deterministic environment behavior through explicit seeding,
2. session-isolated API state for concurrent use,
3. a baseline inference runner,
4. a minimal Hugging Face TRL training pipeline and Colab notebook artifact,
5. reward-improvement artifacts for before/after evidence.

## What changed for submission hardening

- Added deterministic seed plumbing in `src/environment.py` and `inference.py`.
- Added session-based environment management in `server/app.py` via `session_id`.
- Added training assets in `training/`:
  - `training/trl_sft_training.py`
  - `training/colab_trl_sft_notebook.ipynb`
  - `training/evaluate_reward_improvement.py`
- Added evidence artifacts in `artifacts/`:
  - `reward_progression.csv`
  - `reward_progression.json`
  - `reward_progression.svg`
  - `trl_sft_training_metrics.json`
- Aligned dependency definitions across `pyproject.toml`, `requirements.txt`, and `setup.py`.

## Environment API

### Endpoints

- `POST /reset` (or `/reset/`)
- `POST /step`
- `POST /state`
- `POST /grade`
- `GET /health`

### Sessionized behavior

`/reset` now returns `session_id`. Use that `session_id` for subsequent calls.

Reset request sources:
- `session_id`: query param, `x-session-id` header, or JSON body
- `task_id`: query param or JSON body
- `seed`: query param or JSON body

If `session_id` is omitted in `/reset`, a UUID is generated and returned.

## Determinism and reproducibility

- `DataCleaningEnv` supports `reset(task_id=..., seed=...)`.
- Python and NumPy RNGs are explicitly seeded.
- `inference.py` uses `INFERENCE_SEED` and sets `TEMPERATURE=0.0`.

Example deterministic run:

```bash
INFERENCE_SEED=2026 python inference.py | tail -n 1
# [SUMMARY] average_score=0.672656
```

Running the same command twice with the same seed yields the same summary.

## Training deliverables (TRL + Colab)

### Local TRL script

```bash
pip install -r requirements.txt
pip install trl transformers datasets accelerate torch
python training/trl_sft_training.py
```

Outputs:
- `artifacts/trl_sft_training_metrics.json`
- `artifacts/trl_sft_checkpoint/` (ignored in git via `.gitignore`)

### Colab notebook

- `training/colab_trl_sft_notebook.ipynb`

Use this notebook to reproduce a minimal TRL SFT run in Colab and export metrics.

## Reward-improvement evidence

Generate evidence:

```bash
python training/evaluate_reward_improvement.py
```

Generated artifacts:
- `artifacts/reward_progression.csv`
- `artifacts/reward_progression.json`
- `artifacts/reward_progression.svg`

Current staged progression in `reward_progression.csv`:

| stage | average_score |
| --- | --- |
| baseline | 0.655990 |
| mid | 0.958862 |
| trained | 0.990000 |

## Setup

```bash
git clone https://github.com/redhatsam09/scaler-final.git
cd scaler-final
pip install -r requirements.txt
pip install -e .
```

## Run locally

Server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Baseline inference:

```bash
python inference.py
```

OpenEnv validation:

```bash
openenv validate
```

## Submission links (fill with your final public URLs)

- Hugging Face Space URL: `https://huggingface.co/spaces/samdutta123/scaler-final-openenv`
- Live API base URL: `https://samdutta123-scaler-final-openenv.hf.space`
- Colab notebook URL: `REPLACE_WITH_FINAL_COLAB_URL`
- Mini-blog URL: `REPLACE_WITH_FINAL_BLOG_URL`
- Mini-video URL (<2 min): `REPLACE_WITH_FINAL_VIDEO_URL`

> Do not mark submission as fully ready until the placeholders above are replaced with live public links.
