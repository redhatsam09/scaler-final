# Deployment Guide (GitHub + Hugging Face Space)

This project supports token-only Hugging Face deployment.

## 1) Run local quality checks

```bash
cd /workspaces/scaler-final
./scripts/run_local_checks.sh
```

## 2) Commit and push to GitHub

```bash
cd /workspaces/scaler-final
git add .
git commit -m "Finalize production hardening and submission pipeline"
git push origin opus
```

Use your target branch in place of `opus` if needed.

If your Space is connected to GitHub `main`, promote your validated branch:

```bash
cd /workspaces/scaler-final
git push origin opus:main
```

## 3) Deploy to Hugging Face Space using token auth

Required token scope: write access to the Space repository.

```bash
cd /workspaces/scaler-final
export HF_TOKEN="<your_hf_write_token>"
export HF_SPACE_ID="samdutta123/scaler-final-openenv"
./scripts/deploy_hf_space.sh
```

Optional overrides:

```bash
export HF_TARGET_BRANCH="main"
export HF_SPACE_ID="<owner>/<space-name>"
```

## 4) Verify live API endpoints

```bash
curl -sS https://samdutta123-scaler-final-openenv.hf.space/health
curl -sS -X POST 'https://samdutta123-scaler-final-openenv.hf.space/reset' \
	-H 'Content-Type: application/json' \
	-d '{"task_id":"task_enterprise_orchestration","seed":2026}'
curl -sS https://samdutta123-scaler-final-openenv.hf.space/openapi.json | grep '"/close"'
```

Expected signals after the latest deployment:
- `/health` includes `active_sessions` and `session_ttl_seconds`
- OpenAPI includes `/close`

## 5) Final README links for submission

Ensure the README contains reachable links for:
- Hugging Face Space
- Live API
- Colab notebook
- Mini-blog or short video (or slide deck)
