#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required. Export a Hugging Face write token and retry."
  exit 1
fi

HF_SPACE_ID="${HF_SPACE_ID:-samdutta123/scaler-final-openenv}"
TARGET_BRANCH="${HF_TARGET_BRANCH:-main}"
SPACE_HOST="${HF_SPACE_ID/\//-}.hf.space"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Working tree is dirty. Commit changes before deployment."
  exit 1
fi

echo "Deploying HEAD to Hugging Face Space: ${HF_SPACE_ID} (${TARGET_BRANCH})"
git push "https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_ID}" "HEAD:${TARGET_BRANCH}"

echo "Waiting for Space build to pick up changes..."

echo "Health endpoint:" 
curl -fsS "https://${SPACE_HOST}/health" | cat

echo "Demo endpoint:" 
echo "https://${SPACE_HOST}/demo"
