#!/bin/bash
# =============================================================================
# SGP-Tribe3 Deployment Script
# Run this on your LOCAL machine to push the Space to HuggingFace
# =============================================================================
#
# BEFORE RUNNING:
#   1. Make sure you've revoked the old token at https://huggingface.co/settings/tokens
#   2. Generate a new write-access token and set it below (or export it first)
#   3. Make sure git and git-lfs are installed: sudo apt install git git-lfs
#   4. Check your HuggingFace username at https://huggingface.co/settings/profile
#
# USAGE:
#   export HF_TOKEN=hf_your_new_token_here
#   export HF_USERNAME=your_username_here
#   bash deploy.sh
# =============================================================================

set -e  # Exit on any error

# ── Configuration ─────────────────────────────────────────────────────────────
HF_USERNAME="${HF_USERNAME:-REPLACE_WITH_YOUR_USERNAME}"
HF_TOKEN="${HF_TOKEN:-REPLACE_WITH_YOUR_TOKEN}"
SPACE_NAME="sgp-tribe3"
SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
REPO_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "============================================"
echo "  SGP-Tribe3 Deployment"
echo "  Space: ${SPACE_URL}"
echo "============================================"

# ── Step 1: Install huggingface_hub CLI if needed ─────────────────────────────
echo ""
echo "[1/6] Checking huggingface-cli..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub --quiet
fi

# ── Step 2: Create the Space on HuggingFace ───────────────────────────────────
echo ""
echo "[2/6] Creating HuggingFace Space (if it doesn't exist)..."
python3 - <<EOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token="${HF_TOKEN}")
try:
    repo_info = api.repo_info(
        repo_id="${HF_USERNAME}/${SPACE_NAME}",
        repo_type="space"
    )
    print(f"  Space already exists: {repo_info.id}")
except Exception:
    print("  Creating new Space...")
    create_repo(
        repo_id="${SPACE_NAME}",
        repo_type="space",
        space_sdk="docker",
        token="${HF_TOKEN}",
        private=False,
    )
    print(f"  Space created: ${SPACE_URL}")

# Set HF_TOKEN as a Space secret (needed for LLaMA access)
try:
    api.add_space_secret(
        repo_id="${HF_USERNAME}/${SPACE_NAME}",
        key="HF_TOKEN",
        value="${HF_TOKEN}",
    )
    print("  HF_TOKEN secret set in Space")
except Exception as e:
    print(f"  Could not set secret automatically: {e}")
    print("  --> Set it manually at: ${SPACE_URL}/settings")
EOF

# ── Step 3: Initialize git repo ───────────────────────────────────────────────
echo ""
echo "[3/6] Setting up git repository..."
git lfs install --skip-smudge 2>/dev/null || true

DEPLOY_DIR="$(dirname "$0")"
cd "${DEPLOY_DIR}"

if [ ! -d ".git" ]; then
    git init
    git lfs track "*.mp4" "*.wav" "*.pt" "*.bin"
    echo ".gitattributes" >> .gitattributes
fi

# ── Step 4: Configure remote ──────────────────────────────────────────────────
echo ""
echo "[4/6] Configuring git remote..."
git remote remove hf_space 2>/dev/null || true
git remote add hf_space "${REPO_URL}"

# ── Step 5: Commit files ──────────────────────────────────────────────────────
echo ""
echo "[5/6] Committing SGP-Tribe3 files..."
git config user.email "sgp-tribe3@local"
git config user.name "SGP-Tribe3"

git add README.md Dockerfile requirements.txt app.py sgp_parcellation.py stimulus_pipeline.py
git add .gitattributes 2>/dev/null || true

git commit -m "SGP-Tribe3 v1.0: Multimodal brain encoding with 9-node SGP parcellation

- Full video+audio+text inference via TRIBE v2
- Schaefer-200 atlas parcellation to 9 SGP nodes
- Dual-stream (dorsal/ventral) activation separation  
- White matter tract co-activation edge weights
- Co-activation matrix endpoint for Resonance Graph calibration
- REST API: /predict /health /warmup /nodes /tracts /results /coactivation_matrix
" 2>/dev/null || echo "  (Nothing new to commit)"

# ── Step 6: Push to HuggingFace ───────────────────────────────────────────────
echo ""
echo "[6/6] Pushing to HuggingFace Spaces..."
git push hf_space main --force

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "  Space URL: ${SPACE_URL}"
echo "  API URL:   https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo ""
echo "  Next steps:"
echo "  1. Wait ~5-10 min for Space to build (watch logs at Space URL)"
echo "  2. POST to https://${HF_USERNAME}-${SPACE_NAME}.hf.space/warmup"
echo "  3. Wait for model to load (~5 min)"
echo "  4. Test with a local video:"
echo "     python3 stimulus_pipeline.py \\"
echo "       --api https://${HF_USERNAME}-${SPACE_NAME}.hf.space \\"
echo "       --local-video /path/to/your/video.mp4"
echo "============================================"
