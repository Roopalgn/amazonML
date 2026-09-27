"""Launch a SageMaker Training Job for the Amazon ML Challenge pipeline.

Usage:
    python aws/launch_training_job.py --bucket YOUR_BUCKET --prefix amazonml

Prerequisites:
    pip install boto3 sagemaker
    aws configure   (or set AWS_PROFILE / IAM role)

The job uses the SageMaker scikit-learn CPU container as a plain Python
runtime, then installs the challenge-specific requirements at job start.

This is a CPU/string-matching workload. Do not use a GPU instance for the
current pipeline unless the model code changes substantially.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launch(args: argparse.Namespace) -> None:
    import boto3
    import sagemaker
    from sagemaker.sklearn.estimator import SKLearn

    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    role = args.role or sagemaker.get_execution_role()

    # Hyperparameters become environment variables in the container via SM_ prefix.
    # The entrypoint reads these to tune memory and sample sizes.
    environment = {
        "SM_MEMORY":        args.memory,
        "SM_TRAIN_SIZE":    str(args.train_size),
        "SM_TUNE_SIZE":     str(args.tune_size),
        "SM_HOLDOUT_SIZE":  str(args.holdout_size),
        "SM_BATCH_SIZE":    str(args.batch_size),
    }

    job_name = f"amazonml-pipeline-{int(time.time())}"

    estimator = SKLearn(
        framework_version="1.2-1",
        py_version="py3",
        role=role,
        entry_point="train_entrypoint.py",
        source_dir=str(ROOT / "aws"),
        dependencies=[
            str(ROOT / "src"),
            str(ROOT / "requirements-matching.txt"),
            str(ROOT / "data" / "processed"),
        ],
        instance_count=1,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        max_run=args.max_hours * 3600,
        output_path=f"s3://{args.bucket}/{args.prefix}/jobs/",
        code_location=f"s3://{args.bucket}/{args.prefix}/jobs/",
        base_job_name="amazonml-pipeline",
        sagemaker_session=session,
        environment=environment,
    )

    # Input channel: the entire dataset/ directory on S3
    training_input = sagemaker.inputs.TrainingInput(
        s3_data=f"s3://{args.bucket}/{args.prefix}/dataset/",
        s3_data_type="S3Prefix",
        content_type="text/csv",
    )

    print(f"Launching job: {job_name}")
    print(f"Instance:      {args.instance_type}")
    print(f"EBS volume:    {args.volume_size} GB")
    print(f"Memory limit:  {args.memory}")
    print(f"Train size:    {args.train_size:,} queries")
    print(f"Output:        s3://{args.bucket}/{args.prefix}/jobs/")

    estimator.fit(
        inputs={"training": training_input},
        job_name=job_name,
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if args.wait:
        print(f"\nJob complete. Downloading output files...")
        download_outputs(args, job_name, session)
    else:
        print(f"\nJob submitted. Monitor at:")
        print(f"  https://console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs/{job_name}")
        print(f"\nTo download outputs when done:")
        print(f"  python aws/launch_training_job.py --download-job {job_name} --bucket {args.bucket} --prefix {args.prefix}")


def download_outputs(args: argparse.Namespace, job_name: str, session=None) -> None:
    import boto3
    import sagemaker

    if session is None:
        session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))

    s3 = boto3.client("s3", region_name=args.region)
    output_prefix = f"{args.prefix}/jobs/{job_name}/output/output.tar.gz"

    local_tar = ROOT / "output" / "sagemaker_output.tar.gz"
    local_tar.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading s3://{args.bucket}/{output_prefix} ...")
    s3.download_file(args.bucket, output_prefix, str(local_tar))

    import tarfile
    out_dir = ROOT / "output" / "sagemaker_v1"
    out_dir.mkdir(exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(out_dir)

    print(f"Extracted to {out_dir}")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size / 1e6:.1f} MB)")

    # Copy submission files to canonical output/
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        src = out_dir / name
        if src.exists():
            dst = ROOT / "output" / name
            import shutil
            shutil.copy2(src, dst)
            print(f"\nCopied {name} -> output/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket",        required=True, help="S3 bucket name")
    parser.add_argument("--prefix",        default="amazonml")
    parser.add_argument("--region",        default="us-east-1")
    parser.add_argument("--role",          default=None, help="SageMaker IAM role ARN (auto-detected on EC2/SM)")
    parser.add_argument("--instance-type", default="ml.c5.4xlarge",
                        choices=["ml.c5.large","ml.c5.xlarge","ml.c5.2xlarge","ml.c5.4xlarge",
                                 "ml.c5.9xlarge","ml.c5.18xlarge","ml.m5.large","ml.m5.xlarge",
                                 "ml.m5.2xlarge","ml.m5.4xlarge","ml.m5.12xlarge"])
    parser.add_argument("--volume-size",   type=int, default=300, help="Training EBS volume in GB")
    parser.add_argument("--memory",        default="24GB",   help="DuckDB memory limit")
    parser.add_argument("--train-size",    type=int, default=50000, help="Training query count (more = better model)")
    parser.add_argument("--tune-size",     type=int, default=5000)
    parser.add_argument("--holdout-size",  type=int, default=5000)
    parser.add_argument("--batch-size",    type=int, default=50000, help="Scoring batch size")
    parser.add_argument("--max-hours",     type=int, default=6,    help="Job timeout in hours")
    parser.add_argument("--wait",          action="store_true",    help="Block until job finishes and auto-download")
    parser.add_argument("--download-job",  default=None,          help="Download outputs from a completed job name")
    args = parser.parse_args()

    if args.download_job:
        download_outputs(args, args.download_job)
    else:
        launch(args)


if __name__ == "__main__":
    main()
