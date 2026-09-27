"""Launch an EC2 instance that runs the matching pipeline via user data.

The instance receives pre-signed S3 URLs for dataset/code downloads and output
uploads, so it does not need an IAM instance profile or long-lived credentials.
Use instance-initiated shutdown behavior ``terminate`` so a completed run does
not keep billing for compute.
"""

from __future__ import annotations

import argparse
import base64
import json
import textwrap
import time
from pathlib import Path

import boto3


ROOT = Path(__file__).resolve().parents[1]
DATASET_PREFIX = "dataset/6ab10eb3b23ba_student_resource/student_resource/dataset"
FILES = [
    ("train/train_ground_truth.tsv", f"{DATASET_PREFIX}/train/train_ground_truth.tsv"),
    ("train/train_source1.tsv", f"{DATASET_PREFIX}/train/train_source1.tsv"),
    ("train/train_source2.tsv", f"{DATASET_PREFIX}/train/train_source2.tsv"),
    ("train/train_source3.tsv", f"{DATASET_PREFIX}/train/train_source3.tsv"),
    ("test/test_source1.tsv", f"{DATASET_PREFIX}/test/test_source1.tsv"),
    ("test/test_source2.tsv", f"{DATASET_PREFIX}/test/test_source2.tsv"),
    ("test/test_source3.tsv", f"{DATASET_PREFIX}/test/test_source3.tsv"),
]


def latest_ubuntu_ami(ec2_client) -> str:
    response = ec2_client.describe_images(
        Owners=["099720109477"],
        Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(response["Images"], key=lambda image: image["CreationDate"])
    return images[-1]["ImageId"]


def presign_get(s3_client, bucket: str, key: str, expires: int) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def presign_put(s3_client, bucket: str, key: str, expires: int) -> str:
    return s3_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
        HttpMethod="PUT",
    )


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_user_data(args: argparse.Namespace, s3_client) -> str:
    expires = args.url_expiry_hours * 3600
    code_url = presign_get(s3_client, args.bucket, f"{args.prefix}/code/source.tar.gz", expires)
    downloads = [
        {
            "path": local,
            "url": presign_get(s3_client, args.bucket, f"{args.prefix}/dataset/{local}", expires),
        }
        for local, _ in FILES
    ]
    model_downloads = [
        {
            "path": name,
            "url": presign_get(s3_client, args.bucket, f"{args.prefix}/model/{name}", expires),
        }
        for name in ["model.txt", "fast_model.txt", "evaluation.json"]
    ]
    uploads = {
        name: presign_put(s3_client, args.bucket, f"{args.prefix}/ec2/{args.job_name}/{name}", expires)
        for name in [
            "matching_results.tsv",
            "candidate_pairs.tsv",
            "run.json",
            "evaluation.json",
            "training.log",
            "status.json",
        ]
    }
    download_json = json.dumps(downloads)
    model_json = json.dumps(model_downloads)
    upload_json = json.dumps(uploads)
    train_block = ""
    model_dir = "$WORK/features"
    if args.mode == "all":
        train_block = textwrap.dedent(
            f"""\
            echo "=== train $(date -u) ==="
            "$VENV/bin/python" -m src.matching.supervised_pipeline all-train \\
              --data-root "$DATA_ROOT" \\
              --work "$WORK/features" \\
              --output "$WORK/output" \\
              --memory {args.memory} \\
              --train-size {args.train_size} \\
              --tune-size {args.tune_size} \\
              --holdout-size {args.holdout_size} \\
              --batch-size {args.batch_size}
            """
        )
    else:
        model_dir = "$WORK/model"

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -Eeuo pipefail
        exec > >(tee -a /var/log/amazonml-training.log) 2>&1

        WORK=/opt/amazonml
        REPO=$WORK/repo
        DATA_ROOT=$REPO/{DATASET_PREFIX}
        VENV=$WORK/venv
        DOWNLOADS={shell_quote(download_json)}
        MODEL_DOWNLOADS={shell_quote(model_json)}
        UPLOADS={shell_quote(upload_json)}

        upload_file() {{
          local name="$1"
          local path="$2"
          local url
          url=$(python3 - <<PY
        import json
        print(json.loads('''$UPLOADS''')["$name"])
        PY
        )
          if [ -f "$path" ]; then
            curl --fail --silent --show-error -X PUT --upload-file "$path" "$url"
          fi
        }}

        finish() {{
          code=$?
          python3 - <<PY > /tmp/status.json
        import json, time
        print(json.dumps({{"exit_code": $code, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}, indent=2))
        PY
          upload_file status.json /tmp/status.json || true
          upload_file training.log /var/log/amazonml-training.log || true
          shutdown -h now || true
        }}
        trap finish EXIT

        echo "=== bootstrap $(date -u) ==="
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip curl tar
        mkdir -p "$REPO" "$DATA_ROOT/train" "$DATA_ROOT/test" "$WORK/output" "$WORK/features"

        echo "=== download code ==="
        curl --fail --location --retry 5 --retry-delay 10 {shell_quote(code_url)} -o "$WORK/source.tar.gz"
        tar -xzf "$WORK/source.tar.gz" -C "$REPO"

        echo "=== download dataset ==="
        python3 - <<'PY'
        import json, subprocess
        from pathlib import Path
        root = Path("/opt/amazonml/repo/{DATASET_PREFIX}")
        downloads = json.loads({shell_quote(download_json)})
        for item in downloads:
            dest = root / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {{item['path']}}", flush=True)
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "10", item["url"], "-o", str(dest)], check=True)
        PY

        if [ "{args.mode}" = "predict-only" ]; then
          echo "=== download model ==="
          mkdir -p "$WORK/model"
          python3 - <<'PY'
        import json, subprocess
        from pathlib import Path
        root = Path("/opt/amazonml/model")
        downloads = json.loads({shell_quote(model_json)})
        for item in downloads:
            dest = root / item["path"]
            print(f"downloading model {{item['path']}}", flush=True)
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "10", item["url"], "-o", str(dest)], check=True)
        PY
        fi

        echo "=== install python deps ==="
        python3 -m venv "$VENV"
        "$VENV/bin/python" -m pip install --upgrade pip
        "$VENV/bin/python" -m pip install -r "$REPO/requirements-matching.txt"

        cd "$REPO"
        {train_block}

        echo "=== predict $(date -u) ==="
        "$VENV/bin/python" -m src.matching.supervised_pipeline all-test \
          --data-root "$DATA_ROOT" \
          --work "$WORK/features" \
          --model-dir "{model_dir}" \
          --output "$WORK/output" \
          --memory {args.memory} \
          --batch-size {args.batch_size}

        echo "=== upload outputs $(date -u) ==="
        upload_file matching_results.tsv "$WORK/output/matching_results.tsv"
        upload_file candidate_pairs.tsv "$WORK/output/candidate_pairs.tsv"
        upload_file run.json "$WORK/output/run.json"
        upload_file evaluation.json "{model_dir}/evaluation.json"
        echo "=== complete $(date -u) ==="
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="amazonml")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="m5.4xlarge")
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--volume-size", type=int, default=300)
    parser.add_argument("--memory", default="24GB")
    parser.add_argument("--mode", choices=["all", "predict-only"], default="all")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--tune-size", type=int, default=5000)
    parser.add_argument("--holdout-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=30000)
    parser.add_argument("--url-expiry-hours", type=int, default=48)
    parser.add_argument("--job-name", default=f"amazonml-ec2-{int(time.time())}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    ec2_client = session.client("ec2")
    s3_client = session.client("s3")
    ami_id = latest_ubuntu_ami(ec2_client)
    user_data = build_user_data(args, s3_client)
    encoded_size = len(base64.b64encode(user_data.encode("utf-8")))
    if encoded_size > 16384:
        raise ValueError(f"user data is too large after base64 encoding: {encoded_size} bytes")

    response = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        SubnetId=args.subnet_id,
        SecurityGroupIds=[args.security_group_id],
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": args.volume_size,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        UserData=user_data,
        DryRun=args.dry_run,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": args.job_name},
                    {"Key": "Project", "Value": "amazonml"},
                ],
            }
        ],
    )
    instance = response["Instances"][0]
    print(json.dumps({
        "job_name": args.job_name,
        "instance_id": instance["InstanceId"],
        "instance_type": args.instance_type,
        "ami_id": ami_id,
        "output_prefix": f"s3://{args.bucket}/{args.prefix}/ec2/{args.job_name}/",
    }, indent=2))


if __name__ == "__main__":
    main()
