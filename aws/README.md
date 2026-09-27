# AWS training runbook

This pipeline can run on AWS, and it is a good fit when the local laptop is too slow. It is mostly CPU-bound string matching and LightGBM scoring, so use a CPU instance with enough RAM and disk. A GPU is not useful for the current code.

I can prepare and launch the job once this machine has AWS access, but do not paste AWS secret keys into chat. Use one of these safe access paths:

- Run from AWS CloudShell, where AWS CLI is already authenticated.
- Install AWS CLI locally and run `aws configure sso`, then let me use the configured profile.
- Open an EC2/SageMaker terminal that already has an IAM role attached.

This repo has a project-local AWS CLI available through `.venv`, so a system-wide AWS install is not required.

## Recommended setup

Use SageMaker training if you want the least manual server work. Use a CPU instance with at least 32 GB RAM, preferably 64 GB or more for full-test iteration.

Good first choices:

- `ml.c5.9xlarge`: fast CPU, 72 GB RAM.
- `ml.m5.12xlarge`: more memory headroom, useful if retrieval spills heavily.
- `ml.c5.18xlarge`: faster but can burn credits faster.

Set the attached training volume to at least `300` GB. Use `500` GB if you run multiple experiments or keep intermediate DuckDB files.

Check the live AWS price in your region before launching. The challenge credits are useful, but not compulsory; AWS will still bill normal account spend after credits are exhausted or if credits do not apply.

## One-time AWS prerequisites

Create or choose an S3 bucket:

```bash
.\.venv\Scripts\python.exe -m awscli s3 mb s3://YOUR-BUCKET-NAME --region us-east-1
```

Create a SageMaker execution role in the AWS Console:

- Trusted service: SageMaker
- Permissions: access to the S3 bucket above and SageMaker training
- Copy the role ARN, for example `arn:aws:iam::123456789012:role/SageMakerExecutionRole`

Use least-privilege S3 access if possible. `AmazonS3FullAccess` is convenient for a short hackathon run, but bucket-scoped access is safer.

## Upload dataset

From the repo root:

```bash
.\.venv\Scripts\python.exe aws/upload_data.py --bucket YOUR-BUCKET-NAME --prefix amazonml
```

This uploads the official train/test TSV files. If the dataset is already in S3 and only code changed:

```bash
.\.venv\Scripts\python.exe aws/upload_data.py --bucket YOUR-BUCKET-NAME --prefix amazonml --skip-dataset
```

## Launch training

Starter run:

```bash
.\.venv\Scripts\python.exe aws/launch_training_job.py \
  --bucket YOUR-BUCKET-NAME \
  --prefix amazonml \
  --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
  --instance-type ml.c5.9xlarge \
  --volume-size 300 \
  --memory 48GB \
  --train-size 50000 \
  --tune-size 5000 \
  --holdout-size 5000 \
  --batch-size 50000 \
  --wait
```

If the run is too slow or spills heavily, use `ml.m5.12xlarge`, `--volume-size 500`, and `--memory 120GB`.

If the run fails before prediction, do not submit anything. Fix the job and rerun. We only spend one of the five daily challenge submissions after the output validates locally.

## Download outputs

With `--wait`, the launcher downloads output automatically. Without `--wait`, download a completed job later:

```bash
.\.venv\Scripts\python.exe aws/launch_training_job.py \
  --download-job amazonml-pipeline-TIMESTAMP \
  --bucket YOUR-BUCKET-NAME \
  --prefix amazonml
```

Expected local files:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `output/sagemaker_v1/evaluation.json`
- `output/sagemaker_v1/run.json`

## Validate before submitting

Run the official validator:

```bash
python dataset/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

Only submit if validation passes and `run.json` looks sensible:

- all test Source1 rows are present
- candidate pairs are the exact scored set
- matched IDs are a subset of candidate IDs
- the run did not crash or stop early

## Credit discipline

Do not run many blind AWS jobs. The best loop is:

1. Measure local holdout score and slice scores.
2. Make one targeted retrieval/model change.
3. Run SageMaker once for the full test output.
4. Validate locally.
5. Submit only the best candidate for that iteration window.

This matters because the challenge allows only five submissions per day. AWS can make experiments faster, but leaderboard feedback is still the scarce resource.
