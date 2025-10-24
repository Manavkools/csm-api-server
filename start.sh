#!/usr/bin/env bash
set -euo pipefail

# start.sh - ensure model present (download from Hugging Face if missing) then start api.py
# Env vars:
#  - HF_TOKEN     (optional if model already present or mounted) -> Hugging Face token for private downloads
#  - MODEL_DIR    (default: /workspace/csm-1b) -> where to place / look for the model
#  - INFERENCE_CMD (if contains {model_dir}, we will substitute MODEL_DIR into it)
#  - WORK_DIR     (optional) -> used by api.py
#  - PORT         -> port used by api.py

MODEL_DIR="${MODEL_DIR:-/workspace/csm-1b}"
HF_REPO_ID="${HF_REPO_ID:-sesame/csm-1b}"   # Hugging Face repo to download from
# INFERENCE_CMD env expected to include placeholder {model_dir}, {input}, {output}
: "${INFERENCE_CMD:=python csm/examples/transcribe.py --model {model_dir} --input {input} --output {output}}"

echo "==== start.sh ===="
echo "MODEL_DIR: $MODEL_DIR"
echo "HF_REPO_ID: $HF_REPO_ID"
echo "WORK_DIR: ${WORK_DIR:-/tmp/csm_api}"
echo "PORT: ${PORT:-8000}"

# function to check model validity (basic)
model_has_weights() {
  # quick heuristic: check for at least one of common filetypes
  if [ -d "$MODEL_DIR" ] && (ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || ls "$MODEL_DIR"/*.pt >/dev/null 2>&1 || [ -f "$MODEL_DIR/config.json" ] ); then
    return 0
  fi
  return 1
}

if model_has_weights; then
  echo "Model appears present in ${MODEL_DIR} — skipping download."
else
  echo "Model not found at ${MODEL_DIR}."
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Automatic download from Hugging Face requires HF_TOKEN for large/models with gated access."
    echo "If the model is public and no token is needed, set HF_TOKEN to an empty string or ignore this message."
  fi

  # Use a small python snippet to download using huggingface_hub.snapshot_download
  echo "Attempting to download model ${HF_REPO_ID} -> ${MODEL_DIR} ..."
  python - <<PY
import os, sys
from huggingface_hub import snapshot_download
model_dir = os.environ.get("MODEL_DIR", "$MODEL_DIR")
repo_id = os.environ.get("HF_REPO_ID", "$HF_REPO_ID")
token = os.environ.get("HF_TOKEN", None)
print("snapshot_download(repo_id=%s, cache_dir=%s)" % (repo_id, model_dir))
# snapshot_download will create a snapshot under cache_dir/..., to put contents directly into MODEL_DIR we can pass cache_dir and then move.
# We'll download to a temp cache and then move contents to model_dir
from pathlib import Path
import shutil, tempfile
tmp = tempfile.mkdtemp(prefix="hf_download_")
try:
    path = snapshot_download(repo_id=repo_id, cache_dir=tmp, repo_type="model", token=token)
    # snapshot_download returns the path to the folder with the files (e.g., tmp/{repo_hash})
    src = Path(path)
    dst = Path(model_dir)
    dst.mkdir(parents=True, exist_ok=True)
    # copy files from src into dst
    for item in src.iterdir():
        # copy files and dirs
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)
    print("Model downloaded into:", dst)
except Exception as e:
    print("ERROR: snapshot_download failed:", e, file=sys.stderr)
    sys.exit(2)
finally:
    # best-effort cleanup tmp
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
PY

  if model_has_weights; then
    echo "Model successfully downloaded into ${MODEL_DIR}."
  else
    echo "ERROR: model download finished but model files not found in ${MODEL_DIR}."
    echo "List /workspace:"
    ls -la /workspace || true
    exit 1
  fi
fi

# Render INFERENCE_CMD: replace {model_dir} placeholder with the real path
# Keep placeholders {input} and {output} intact for api.py usage
RENDERED_CMD="$(python - <<PY
import os, sys
cmd = os.environ.get("INFERENCE_CMD", "")
cmd = cmd.replace("{model_dir}", os.environ.get("MODEL_DIR", "$MODEL_DIR"))
print(cmd)
PY
)"
export INFERENCE_CMD="$RENDERED_CMD"

echo "Final INFERENCE_CMD: $INFERENCE_CMD"

# Ensure WORK_DIR exists
mkdir -p "${WORK_DIR:-/tmp/csm_api}"

# Launch api.py (exec so container PID 1 corresponds to uvicorn process inside api.py)
echo "Starting api.py..."
exec python api.py
