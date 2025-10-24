#!/usr/bin/env python3
import os
import json
import uuid
import shutil
import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ----------------- Config -----------------
# Provide a default command that users can override at deploy time.
# Must contain {input} and {output}
INFERENCE_CMD = os.getenv(
    "INFERENCE_CMD",
    # EXAMPLE (you will override this with the real CSM command on Runpod):
    # "python csm/examples/transcribe.py --model /workspace/csm-1b --input {input} --output {output}"
    "echo '{\"ok\": true, \"note\": \"Set INFERENCE_CMD to your CSM transcribe command\"}' > {output}"
)

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/csm_api"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# -------------- FastAPI app --------------
app = FastAPI(title="Sesame CSM API", version="1.0.0")

class HealthResp(BaseModel):
    status: str
    inference_cmd: str

@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp(status="ok", inference_cmd=INFERENCE_CMD)

def run_inference_cli(input_path: str, output_path: str, timeout: int = 3600) -> dict:
    """
    Run the CLI defined by INFERENCE_CMD.
    INFERENCE_CMD must contain {input} and {output} placeholders.
    """
    if "{input}" not in INFERENCE_CMD or "{output}" not in INFERENCE_CMD:
        raise RuntimeError("INFERENCE_CMD must contain {input} and {output} placeholders")

    cmd = INFERENCE_CMD.format(input=input_path, output=output_path)
    logger.info(f"Running inference command: {cmd}")

    try:
        completed = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        logger.info(f"[stdout] {completed.stdout[:2000]}")
        logger.info(f"[stderr] {completed.stderr[:2000]}")

        if completed.returncode != 0:
            raise RuntimeError(
                f"Inference command failed (rc={completed.returncode}): {completed.stderr}"
            )

        # Prefer JSON file written by command
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    # Not strict JSON? fallback to stdout as opaque text
                    return {"stdout": completed.stdout}

        # Or try to parse stdout as JSON
        try:
            return json.loads(completed.stdout)
        except Exception:
            return {"stdout": completed.stdout}

    except subprocess.TimeoutExpired:
        raise RuntimeError("Inference command timed out")
    except Exception as e:
        raise

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Upload an audio file (form field name: 'file').
    The server writes it to a temp dir, runs INFERENCE_CMD, and returns JSON.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if file.content_type and not file.content_type.startswith("audio"):
        # still allow if content-type is missing; only warn if wrong
        logger.warning(f"Unexpected content-type: {file.content_type}")

    req_id = uuid.uuid4().hex
    req_dir = WORK_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".wav"
    input_path = str(req_dir / f"input{suffix}")
    output_path = str(req_dir / "out.json")

    try:
        # save upload
        with open(input_path, "wb") as f:
            f.write(await file.read())

        if Path(input_path).stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # run CLI
        try:
            result = run_inference_cli(input_path=input_path, output_path=output_path)
        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        return JSONResponse(result)

    finally:
        # best-effort cleanup
        try:
            shutil.rmtree(req_dir)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info"
    )
