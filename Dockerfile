# ============================================================
# Dockerfile for csm-api-server (Sesame + FastAPI wrapper)
# ============================================================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -ms /bin/bash csmuser
USER csmuser
WORKDIR /home/csmuser/app

# Copy application files (assumes you have csm/ cloned in repo root)
COPY --chown=csmuser:csmuser api.py requirements.txt ./
COPY --chown=csmuser:csmuser csm ./csm
COPY --chown=csmuser:csmuser start.sh ./start.sh

# Python packages (huggingface_hub is required for model download)
RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt ; fi
# ensure huggingface_hub available for snapshot_download; install safely
RUN python - <<PYCODE
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
PYCODE

ENV PORT=8000
EXPOSE 8000

# default model path (can be overridden in Runpod UI env)
ENV MODEL_DIR=/workspace/csm-1b
ENV INFERENCE_CMD="python csm/examples/transcribe.py --model {model_dir} --input {input} --output {output}"
ENV WORK_DIR="/tmp/csm_api"

# ensure start.sh executable and run it
RUN chmod +x ./start.sh

ENTRYPOINT ["./start.sh"]
