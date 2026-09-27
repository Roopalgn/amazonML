"""Run the prepared v2 training experiment on a self-terminating EC2 instance."""

import argparse
import json
from pathlib import Path
import tempfile
import time

import boto3
from botocore.exceptions import ClientError

from aws.launch_ec2_training import latest_ubuntu_ami,presign_get,presign_put,shell_quote
from aws.upload_data import build_code_tarball


def user_data(code_url,downloads,uploads,threads,hours):
    puts='\n'.join(f'  {shell_quote(name)}) url={shell_quote(url)} ;;' for name,url in uploads.items())
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
WORK=/opt/amazonml
mkdir -p "$WORK"
exec > >(tee -a "$WORK/training.log") 2>&1
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
 printf '{{"exit_code":%s}}\n' "$rc" > "$WORK/status.json"
 upload_file status.json "$WORK/status.json" || true
 upload_file training.log "$WORK/training.log" || true
 shutdown -h now || true
}}
trap finish EXIT
systemd-run --unit=amazonml-deadline --on-active={hours}h /sbin/shutdown -h now
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv curl tar libgomp1
mkdir -p "$WORK/repo" "$WORK/archive" "$WORK/legacy" "$WORK/model"
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
(while true; do sleep 60; upload_file training.log "$WORK/training.log" || true; done) &
echo 'Starting v2 feature extraction and training'
"$WORK/venv/bin/python" -m src.matching.train_archive --archive "$WORK/archive" --legacy "$WORK/legacy" --work "$WORK/model" --threads {threads}
upload_file model.txt "$WORK/model/model.txt"
upload_file evaluation.json "$WORK/model/evaluation.json"
upload_file aliases.json "$WORK/model/aliases.json"
upload_file weights.json "$WORK/model/weights.json"
echo 'Training complete; artifacts uploaded'
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bucket',required=True)
    parser.add_argument('--region',default='us-east-1')
    parser.add_argument('--prefix',default='amazonml')
    parser.add_argument('--work',type=Path,default=Path('data/features/resolution_v2'))
    parser.add_argument('--instance-type',default='r7i.xlarge')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--subnet-id',required=True)
    parser.add_argument('--security-group-id',required=True)
    parser.add_argument('--max-hours',type=int,default=3)
    parser.add_argument('--job-name',default=f'resolution-v2-{int(time.time())}')
    parser.add_argument('--reuse-uploaded-inputs',action='store_true',
        help='Reuse existing S3 input objects for this job name when size matches.')
    args=parser.parse_args()
    if not 1<=args.max_hours<=6 or not 1<=args.threads<=16:
        raise ValueError('Invalid runtime or thread limit')
    inputs={'archive/pairs.parquet':args.work/'export/pairs.parquet',
            'archive/queries.json':args.work/'export/queries.json',
            'archive/aliases.json':args.work/'aliases.json',
            'archive/weights.json':args.work/'weights.json'}
    for path in inputs.values():
        if not path.is_file() or path.stat().st_size==0:
            raise ValueError(f'Missing training input: {path}')
    if not (args.work/'export/retrieval_audit.json').exists():
        raise ValueError('Training evidence export has not completed')
    session=boto3.Session(region_name=args.region)
    s3=session.client('s3')
    ec2=session.client('ec2')
    prefix=f'{args.prefix}/experiments/{args.job_name}'
    expiry=(args.max_hours+2)*3600
    downloads=[]
    for relative,path in inputs.items():
        key=f'{prefix}/{relative}'
        size=path.stat().st_size
        uploaded=False
        if args.reuse_uploaded_inputs:
            try:
                head=s3.head_object(Bucket=args.bucket,Key=key)
                uploaded=head.get('ContentLength')==size
            except ClientError as exc:
                if exc.response.get('Error',{}).get('Code') not in ('404','NoSuchKey','NotFound'):
                    raise
        if uploaded:
            print(f'Reusing {relative} ({size:,} bytes)',flush=True)
        else:
            print(f'Uploading {relative} ({size:,} bytes)',flush=True)
            s3.upload_file(str(path),args.bucket,key)
        downloads.append({'path':relative,'url':presign_get(s3,args.bucket,key,expiry)})
    for filename in ('fast_model.txt','evaluation.json'):
        downloads.append({'path':'legacy/'+filename,'url':presign_get(s3,args.bucket,f'{args.prefix}/model/{filename}',expiry)})
    with tempfile.TemporaryDirectory() as directory:
        code=Path(directory)/'source.tar.gz'
        build_code_tarball(code)
        s3.upload_file(str(code),args.bucket,f'{prefix}/code.tar.gz')
    code_url=presign_get(s3,args.bucket,f'{prefix}/code.tar.gz',expiry)
    uploads={name:presign_put(s3,args.bucket,f'{prefix}/output/{name}',expiry)
             for name in ('model.txt','evaluation.json','aliases.json','weights.json','training.log','status.json')}
    script=user_data(code_url,downloads,uploads,args.threads,args.max_hours)
    if len(script.encode())>16384:
        raise ValueError('EC2 user data exceeds 16 KiB')
    response=ec2.run_instances(ImageId=latest_ubuntu_ami(ec2),InstanceType=args.instance_type,
        MinCount=1,MaxCount=1,SubnetId=args.subnet_id,SecurityGroupIds=[args.security_group_id],
        InstanceInitiatedShutdownBehavior='terminate',MetadataOptions={'HttpTokens':'required'},
        BlockDeviceMappings=[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':80,'VolumeType':'gp3','DeleteOnTermination':True}}],
        UserData=script,TagSpecifications=[{'ResourceType':'instance','Tags':[{'Key':'Name','Value':args.job_name},{'Key':'Project','Value':'amazonml'}]}])
    result={'job_name':args.job_name,'instance_id':response['Instances'][0]['InstanceId'],
            'instance_type':args.instance_type,'max_hours':args.max_hours,
            'output_prefix':f's3://{args.bucket}/{prefix}/output/'}
    (args.work/'aws_job.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
