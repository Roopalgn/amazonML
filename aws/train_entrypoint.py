"""SageMaker Training Job entrypoint.

SageMaker mounts input channels at /opt/ml/input/data/<channel_name>/
and expects output at /opt/ml/output/ and model artifacts at /opt/ml/model/.

Channel layout expected:
    /opt/ml/input/data/training/train/       <- train TSVs + ground truth
    /opt/ml/input/data/training/test/        <- test TSVs

This script runs the full pipeline:
    1. prepare   - normalize all source files into DuckDB
    2. retrieve  - blocking / candidate retrieval
    3. train     - fit LightGBM model
    4. predict   - score test candidates, write output files

Final submission files are copied to /opt/ml/model/ for download.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# SageMaker environment paths
SM_INPUT   = Path("/opt/ml/input/data/training")
SM_OUTPUT  = Path("/opt/ml/output")
SM_MODEL   = Path("/opt/ml/model")
SM_CODE    = Path("/opt/ml/code")   # SageMaker extracts source.tar.gz here

# Where we put working files (fast NVMe scratch on ml.c5 instances)
WORK_DIR   = Path("/tmp/amazonml_work")
OUTPUT_DIR = Path("/tmp/amazonml_output")


def log(msg: str) -> None:
    print(f"[entrypoint] {msg}", flush=True)


def install_deps() -> None:
    # requirements-matching.txt is uploaded as a dependency file
    req = SM_CODE / "requirements-matching.txt"
    if not req.exists():
        req = SM_CODE / "requirements-matching.txt"
    log(f"Installing dependencies from {req}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"],
        check=True
    )


def run_pipeline_stage(stage: str, extra_args: list[str] | None = None) -> None:
    cmd = [
        sys.executable, "-m", "src.matching.supervised_pipeline",
        stage,
        "--data-root", str(SM_INPUT),
        "--work",     str(WORK_DIR),
        "--output",   str(OUTPUT_DIR),
        "--memory",   os.environ.get("SM_MEMORY", "24GB"),
        "--train-size",   os.environ.get("SM_TRAIN_SIZE",   "50000"),
        "--tune-size",    os.environ.get("SM_TUNE_SIZE",    "5000"),
        "--holdout-size", os.environ.get("SM_HOLDOUT_SIZE", "5000"),
        "--batch-size",   os.environ.get("SM_BATCH_SIZE",   "50000"),
    ]
    if extra_args:
        cmd.extend(extra_args)
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(SM_CODE))


def copy_outputs() -> None:
    """Copy submission files and model to /opt/ml/model/ for artifact download."""
    SM_MODEL.mkdir(parents=True, exist_ok=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv", "run.json"):
        src = OUTPUT_DIR / name
        if src.exists():
            shutil.copy2(src, SM_MODEL / name)
            log(f"Copied {name} to model dir")

    # Copy trained model files
    for name in ("model.txt", "fast_model.txt", "evaluation.json"):
        src = WORK_DIR / name
        if src.exists():
            shutil.copy2(src, SM_MODEL / name)
            log(f"Copied {name} to model dir")

    log(f"Artifacts in {SM_MODEL}:")
    for f in sorted(SM_MODEL.iterdir()):
        log(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    log("=== Amazon ML Challenge - Full Pipeline ===")
    log(f"Instance type hint: {os.environ.get('SM_NUM_CPUS', '?')} CPUs")
    log(f"Work dir: {WORK_DIR}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # SageMaker Script Mode puts our code at /opt/ml/code/.
    # The 'dependencies' list (src/, data/processed/) land there too.
    # Add it to sys.path so `src.*` imports resolve.
    sys.path.insert(0, str(SM_CODE))
    os.chdir(str(SM_CODE))  # Make relative imports consistent

    install_deps()

    # Stage 1: normalize train data + select queries
    log("=== Stage: prepare (train) ===")
    run_pipeline_stage("prepare", ["--split", "train"])

    # Stage 2: retrieve train candidates
    log("=== Stage: retrieve (train) ===")
    run_pipeline_stage("retrieve", ["--split", "train"])

    # Stage 3: train LightGBM
    log("=== Stage: train ===")
    run_pipeline_stage("train")

    # Stage 4: normalize test data + select queries
    log("=== Stage: prepare (test) ===")
    run_pipeline_stage("prepare", ["--split", "test"])

    # Stage 5: retrieve test candidates
    log("=== Stage: retrieve (test) ===")
    run_pipeline_stage("retrieve", ["--split", "test"])

    # Stage 6: score test candidates -> output files
    log("=== Stage: predict ===")
    run_pipeline_stage("predict", ["--model-dir", str(WORK_DIR)])

    copy_outputs()
    log("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
