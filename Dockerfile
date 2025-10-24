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
    && rm -rf /var/lib/apt/lists/*

# optional: non-root user
RUN useradd -ms /bin/bash csmuser
USER csmuser
WORKDIR /home/csmuser/app

COPY --chown=csmuser:csmuser api.py requirements.txt ./
COPY --chown=csmuser:csmuser csm ./csm

RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt ; fi
RUN if [ -f csm/requirements.txt ]; then pip install -r csm/requirements.txt ; fi

ENV PORT=8000
EXPOSE 8000

# Command pattern: replace with your model path at runtime
ENV INFERENCE_CMD="python csm/examples/transcribe.py --model /workspace/csm-1b --input {input} --output {output}"
ENV WORK_DIR="/tmp/csm_api"

CMD ["python", "api.py"]
