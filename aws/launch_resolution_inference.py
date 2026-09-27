"""Run v2 full-test inference on a self-terminating EC2 instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import boto3

from aws.launch_ec2_training import FILES, latest_ubuntu_ami, presign_get, presign_put, shell_quote
from aws.upload_data import build_code_tarball


def user_data(code_url, downloads, uploads, threads, hours, memory, batch_size, margin,
              skip_legacy, route_families):
    puts = "\n".join(f"  {shell_quote(name)}) url={shell_quote(url)} ;;" for name, url in uploads.items())
    deadline = "" if hours == 0 else f"systemd-run --unit=amazonml-inference-deadline --on-active={hours}h /sbin/shutdown -h now"
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
WORK=/opt/amazonml
mkdir -p "$WORK"
exec > >(tee -a "$WORK/inference.log") 2>&1
upload_file() {{
 local name="$1" path="$2" url
 case "$name" in
{puts}
 esac
 if [ -f "$path" ]; then curl --fail --silent --show-error --retry 2 -X PUT --upload-file "$path" "$url"; fi
}}
finish() {{
 local rc=$?
 trap - EXIT
 printf '{{"exit_code":%s}}\\n' "$rc" > "$WORK/status.json"
 upload_file status.json "$WORK/status.json" || true
 upload_file inference.log "$WORK/inference.log" || true
 shutdown -h now || true
}}
trap finish EXIT
{deadline}
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv curl tar libgomp1 coreutils
mkdir -p "$WORK/repo" "$WORK/data/train" "$WORK/data/test" "$WORK/model" "$WORK/legacy" "$WORK/features" "$WORK/output"
curl --fail --silent --show-error --location --retry 4 {shell_quote(code_url)} -o "$WORK/code.tar.gz"
tar -xzf "$WORK/code.tar.gz" -C "$WORK/repo"
python3 - <<'PY'
import pathlib, subprocess
items={downloads!r}
for item in items:
    path=pathlib.Path('/opt/amazonml') / item['path']
    path.parent.mkdir(parents=True,exist_ok=True)
    print('Downloading '+item['path'],flush=True)
    result=subprocess.run(['curl','--fail','--silent','--show-error','--location','--retry','4',item['url'],'-o',str(path)])
    if result.returncode:
        raise RuntimeError('Download failed: '+item['path'])
PY
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install -q -r "$WORK/repo/requirements-matching.txt"
cd "$WORK/repo"
(while true; do sleep 60; upload_file inference.log "$WORK/inference.log" || true; done) &
export RESOLUTION_SKIP_LEGACY={1 if skip_legacy else 0}
export RESOLUTION_ROUTE_FAMILIES={shell_quote(route_families)}
if [ "$RESOLUTION_SKIP_LEGACY" = "0" ]; then
echo 'Importing baseline candidate pairs'
"$WORK/venv/bin/python" - <<'PY'
import duckdb, pathlib
root=pathlib.Path('/opt/amazonml')
con=duckdb.connect(str(root/'legacy/test.duckdb'))
con.execute("SET memory_limit='2GB'")
con.execute("SET threads={threads}")
con.execute("SET preserve_insertion_order=false")
con.execute("SET temp_directory='/opt/amazonml/legacy/spill'")
con.execute(\"\"\"CREATE TABLE pairs AS
    SELECT source1_entity_id qid, unnest(string_split(candidate_entity_ids, ',')) tid
    FROM read_csv('/opt/amazonml/legacy/candidate_pairs.tsv',delim='\\t',header=true,all_varchar=true)
    WHERE coalesce(candidate_entity_ids,'')!=''
\"\"\")
print('legacy pairs', con.execute('SELECT count(*) FROM pairs').fetchone()[0], flush=True)
con.close()
PY
else
echo 'Skipping baseline candidate import'
fi
echo 'Starting v2 full test inference'
"$WORK/venv/bin/python" -m src.matching.resolution_pipeline all-test --data-root "$WORK/data" --work "$WORK/features" --model-dir "$WORK/model" --legacy "$WORK/legacy" --output "$WORK/output_raw" --memory {memory} --threads {threads} --batch-size {batch_size}
echo 'Resolving target ownership conflicts'
"$WORK/venv/bin/python" -m src.matching.resolve_prediction_conflicts --accepted "$WORK/features/prediction_batches/accepted.tsv" --source1 "$WORK/data/test/test_source1.tsv" --output "$WORK/output/matching_results.tsv" --margin {margin}
cp "$WORK/output_raw/candidate_pairs.tsv" "$WORK/output/candidate_pairs.tsv"
cp "$WORK/output_raw/run.json" "$WORK/output/raw_run.json"
upload_file matching_results.tsv "$WORK/output/matching_results.tsv"
upload_file candidate_pairs.tsv "$WORK/output/candidate_pairs.tsv"
upload_file raw_matching_results.tsv "$WORK/output_raw/matching_results.tsv"
upload_file raw_run.json "$WORK/output/raw_run.json"
upload_file inference.log "$WORK/inference.log"
echo 'Inference complete; artifacts uploaded'
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--prefix", default="amazonml")
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--candidate-key")
    parser.add_argument("--skip-legacy-candidates", action="store_true")
    parser.add_argument("--route-families", default="rare_name,rare_address,number_name,number_address,name_pair")
    parser.add_argument("--instance-type", default="m7i-flex.large")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory", default="5GB")
    parser.add_argument("--batch-size", type=int, default=30000)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--volume-size", type=int, default=300)
    parser.add_argument("--max-hours", type=int, default=3,
        help="Auto-shutdown deadline. Use 0 for no deadline guard; completion/failure still shuts down.")
    parser.add_argument("--url-expiry-hours", type=int,
        help="Presigned URL lifetime. Defaults to max-hours+2, or 48 when max-hours is 0.")
    parser.add_argument("--job-name", default=f"resolution-v2-infer-{int(time.time())}")
    args = parser.parse_args()
    if not 0 <= args.max_hours <= 48:
        raise ValueError("Invalid runtime limit")
    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")
    ec2 = session.client("ec2")
    expiry_hours = args.url_expiry_hours if args.url_expiry_hours is not None else (48 if args.max_hours == 0 else args.max_hours + 2)
    expiry = expiry_hours * 3600
    output_prefix = f"{args.prefix}/experiments/{args.job_name}/output"
    downloads = [
        {"path": f"data/{local}", "url": presign_get(s3, args.bucket, f"{args.prefix}/dataset/{local}", expiry)}
        for local, _ in FILES
    ]
    downloads += [
        {"path": f"model/{name}", "url": presign_get(s3, args.bucket, f"{args.model_prefix.rstrip('/')}/{name}", expiry)}
        for name in ("model.txt", "evaluation.json", "aliases.json")
    ]
    downloads += [
        {"path": f"legacy/{name}", "url": presign_get(s3, args.bucket, f"{args.prefix}/model/{name}", expiry)}
        for name in ("fast_model.txt", "evaluation.json")
    ]
    if not args.skip_legacy_candidates:
        if not args.candidate_key:
            raise ValueError("--candidate-key is required unless --skip-legacy-candidates is set")
        downloads.append({"path": "legacy/candidate_pairs.tsv", "url": presign_get(s3, args.bucket, args.candidate_key, expiry)})
    with tempfile.TemporaryDirectory() as directory:
        code = Path(directory) / "source.tar.gz"
        build_code_tarball(code)
        s3.upload_file(str(code), args.bucket, f"{args.prefix}/experiments/{args.job_name}/code.tar.gz")
    code_url = presign_get(s3, args.bucket, f"{args.prefix}/experiments/{args.job_name}/code.tar.gz", expiry)
    uploads = {
        name: presign_put(s3, args.bucket, f"{output_prefix}/{name}", expiry)
        for name in ("matching_results.tsv", "candidate_pairs.tsv", "raw_matching_results.tsv", "raw_run.json", "inference.log", "status.json")
    }
    script = user_data(code_url, downloads, uploads, args.threads, args.max_hours, args.memory,
                       args.batch_size, args.margin, args.skip_legacy_candidates, args.route_families)
    if len(script.encode()) > 16384:
        raise ValueError("EC2 user data exceeds 16 KiB")
    response = ec2.run_instances(
        ImageId=latest_ubuntu_ami(ec2),
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        SubnetId=args.subnet_id,
        SecurityGroupIds=[args.security_group_id],
        InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "required"},
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {
            "VolumeSize": args.volume_size, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        UserData=script,
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": args.job_name}, {"Key": "Project", "Value": "amazonml"}]}],
    )
    result = {
        "job_name": args.job_name,
        "instance_id": response["Instances"][0]["InstanceId"],
        "instance_type": args.instance_type,
        "output_prefix": f"s3://{args.bucket}/{output_prefix}/",
    }
    Path("data/features/resolution_v2/aws_inference_job.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
