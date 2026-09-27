"""Reproducible disk-backed retrieval, training, evaluation and submission CLI.

Only supplied business data is used. Country restricts retrieval but is never a
closed-vocabulary model feature. Labels are separated by Source-1 entity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import duckdb
import lightgbm as lgb
import numpy as np

from src.data.io import find_dataset_root
from src.eval.macro_f05 import entity_f05
from src.matching.pair_features import FEATURE_NAMES, FEATURE_VERSION, FAST_FEATURE_INDICES, fast_features, pair_features


VERSION = "supervised-v1"
ROOT = Path(__file__).resolve().parents[2]
LEGAL = r"\b(incorporated|corporation|company|limited|private|inc|corp|llc|ltd|pvt|plc|sarl|sas|llp)\b"


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sql_path(path):
    return "'" + str(Path(path).resolve()).replace("\\", "/").replace("'", "''") + "'"


def connect(work, split, memory="1GB"):
    work.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(work / f"{split}.duckdb"))
    con.execute("SET threads=2")
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET temp_directory={sql_path(work / ('spill-' + split))}")
    con.execute("SET preserve_insertion_order=false")
    return con


def norm_sql(column):
    return f"trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce({column}, ''))), '[^\\p{{L}}\\p{{N}}\\p{{M}}]+', ' ', 'g'), ' +', ' ', 'g'))"


def source_signature(paths):
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in paths]


def exists(con, name):
    return bool(con.execute("SELECT 1 FROM information_schema.tables WHERE table_name=?", [name]).fetchone())


def prepare(con, root, split, work):
    paths = [root / split / f"{split}_source{i}.tsv" for i in range(1, 4)]
    signature = {"version": VERSION, "files": source_signature(paths)}
    manifest = work / f"{split}.sources.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("Source or normalization version changed; use a fresh --work directory")
        if exists(con, "qfull") and exists(con, "targets"):
            log(f"Reusing normalized {split} records")
            return
    for table, selected in (("qfull", paths[:1]), ("targets", paths[1:])):
        con.execute(f"DROP TABLE IF EXISTS {table}")
        for i, path in enumerate(selected):
            log(f"Normalizing {path.name}")
            source = f"read_csv({sql_path(path)}, delim='\t', header=true, all_varchar=true)"
            statement = f"CREATE TABLE {table} AS" if i == 0 else f"INSERT INTO {table}"
            con.execute(f"""{statement}
                WITH normalized AS (
                    SELECT entity_id AS id, country, {norm_sql('business_name')} AS n,
                           {norm_sql('business_address')} AS a FROM {source}
                ), cleaned AS (
                    SELECT *, trim(regexp_replace(regexp_replace(n, '{LEGAL}', ' ', 'g'), ' +', ' ', 'g')) AS core,
                           regexp_extract(a, '[0-9]+') AS house FROM normalized
                )
                SELECT id, country, n, a, house,
                    array_to_string(list_sort(list_distinct(string_split(CASE WHEN core='' THEN n ELSE core END, ' '))), ' ') AS ns,
                    array_to_string(list_sort(list_distinct(string_split(a, ' '))), ' ') AS ads,
                    list_sort(list_distinct(list_filter(string_split(core, ' '), x -> length(x)>=3))) AS nt,
                    list_slice(list_reverse_sort(list_transform(
                        list_filter(string_split(a, ' '), x -> length(x)>=4 AND NOT regexp_matches(x, '[0-9]')),
                        x -> struct_pack(size:=length(x), token:=x))), 1, 3) AS at
                FROM cleaned""")
            con.execute("CHECKPOINT")
    manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def select_queries(con, work, split, train_size, tune_size, holdout_size):
    config = {"version": VERSION, "split": split, "train": train_size, "tune": tune_size, "holdout": holdout_size}
    path = work / f"{split}.queries.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Query configuration changed; use a fresh --work directory")
        if exists(con, "queries"):
            return
    if split == "test":
        con.execute("CREATE OR REPLACE TABLE queries AS SELECT *, 'test' AS role FROM qfull")
    else:
        val_path = ROOT / "data/processed/validation_source1_ids.txt"
        dev_path = ROOT / "data/processed/dev_sample_source1_ids.txt"
        for name, ids_path in (("fixed_validation", val_path), ("old_dev", dev_path)):
            con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT column0 AS id FROM read_csv({sql_path(ids_path)}, header=false, all_varchar=true)")
        con.execute(f"""CREATE OR REPLACE TABLE queries AS
            WITH fitting AS (
                SELECT q.*, 'fit' AS role FROM qfull q ANTI JOIN fixed_validation USING(id)
                ORDER BY hash(q.id || 'fit-v1') LIMIT {train_size}
            ), evaluation AS (
                SELECT q.*, row_number() OVER (ORDER BY hash(q.id || 'eval-v1')) AS rn
                FROM qfull q JOIN fixed_validation USING(id) ANTI JOIN old_dev USING(id)
            )
            SELECT * FROM fitting UNION ALL
            SELECT * EXCLUDE(rn), CASE WHEN rn <= {tune_size} THEN 'tune' ELSE 'holdout' END AS role
            FROM evaluation WHERE rn <= {tune_size + holdout_size}
        """)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Query roles: {con.execute('SELECT role, count(*) FROM queries GROUP BY role').fetchall()}")


def key_expression(family):
    if family == "name":
        return "[ns]"
    if family == "address":
        return "[ads]"
    if family == "name_pair":
        return "[" + ",".join(f"CASE WHEN length(nt[{j}])>=3 THEN nt[{i}] || '|' || nt[{j}] END" for i in range(1, 4) for j in range(i + 1, 5)) + "]"
    if family == "house_name":
        return "[" + ",".join(f"CASE WHEN house!='' AND length(nt[{i}])>=4 THEN house || '|' || left(nt[{i}],4) END" for i in range(1, 5)) + "]"
    if family == "house_locality":
        return "[" + ",".join(f"CASE WHEN house!='' THEN house || '|' || left((\"at\"[{i}]).token,8) END" for i in range(1, 4)) + "]"
    if family == "locality_pair":
        return "[" + ",".join(f"CASE WHEN length((\"at\"[{j}]).token)>=4 THEN least((\"at\"[{i}]).token,(\"at\"[{j}]).token) || '|' || greatest((\"at\"[{i}]).token,(\"at\"[{j}]).token) END" for i in range(1, 3) for j in range(i + 1, 4)) + "]"
    raise ValueError(family)


def retrieve(con, work, split):
    completed = work / f"{split}.retrieval.json"
    if completed.exists() and exists(con, "pairs"):
        log("Reusing completed retrieval")
        return
    routes = ["name", "address", "name_pair", "house_name", "house_locality", "locality_pair"]
    for family in routes:
        pair_table = "pairs_" + family
        if exists(con, pair_table):
            continue
        log(f"Retrieving {family} candidates against all reference rows")
        expr = key_expression(family)
        con.execute(f"""CREATE OR REPLACE TABLE qkeys AS
            SELECT DISTINCT id AS qid, country, unnest({expr}) AS key FROM queries""")
        con.execute("DELETE FROM qkeys WHERE key IS NULL OR length(key)<4")
        # Frequency limits apply on the full target corpus, never on labels.
        con.execute(f"""CREATE OR REPLACE TABLE tkeys AS
            SELECT DISTINCT t.id AS tid, t.country, t.key FROM (
                SELECT id, country, unnest({expr}) AS key FROM targets
            ) t SEMI JOIN qkeys q USING(country,key)""")
        limit = 1000 if family in {"name", "address"} else 300
        con.execute(f"""CREATE OR REPLACE TABLE {pair_table} AS
            WITH eligible AS (
                SELECT country,key FROM tkeys GROUP BY country,key HAVING count(*)<={limit}
            ) SELECT DISTINCT q.qid,t.tid FROM tkeys t JOIN eligible USING(country,key)
            JOIN qkeys q USING(country,key)""")
        log(f"{family}: {con.execute(f'SELECT count(*) FROM {pair_table}').fetchone()[0]:,} pairs")
        con.execute("DROP TABLE qkeys")
        con.execute("DROP TABLE tkeys")
        con.execute("CHECKPOINT")
    union = " UNION ALL ".join("SELECT * FROM pairs_" + route for route in routes)
    con.execute(f"CREATE OR REPLACE TABLE pairs AS SELECT DISTINCT qid,tid FROM ({union})")
    total = con.execute("SELECT count(*) FROM pairs").fetchone()[0]
    completed.write_text(json.dumps({"version": VERSION, "pairs": total, "routes": routes}, indent=2))
    log(f"Retrieved {total:,} distinct pairs")


def pair_batches(con, batch_size):
    cursor = con.execute("""SELECT p.qid,p.tid,q.n,q.a,t.n,t.a
        FROM pairs p JOIN queries q ON q.id=p.qid JOIN targets t ON t.id=p.tid""")
    while rows := cursor.fetchmany(batch_size):
        yield rows


def load_truth(con, root):
    path = root / "train/train_ground_truth.tsv"
    rows = con.execute(f"""SELECT source1_entity_id, matched_entity_ids
        FROM read_csv({sql_path(path)}, delim='\t',header=true,all_varchar=true)
        SEMI JOIN queries ON id=source1_entity_id""").fetchall()
    return {qid: set(value.split(',')) if value else set() for qid, value in rows}


def build_training_features(con, work, truth, batch_size):
    directory = work / "train_features"
    manifest = directory / "complete.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["version"] != FEATURE_VERSION:
            raise ValueError("Feature cache version mismatch")
        return
    directory.mkdir(exist_ok=True)
    queries = con.execute("SELECT id,role FROM queries ORDER BY id").fetchall()
    query_indices = {qid: i for i, (qid, _) in enumerate(queries)}
    (directory / "queries.json").write_text(json.dumps(queries))
    count = 0
    for batch_no, rows in enumerate(pair_batches(con, batch_size)):
        features = pair_features(rows)
        labels = np.array([tid in truth[qid] for qid, tid, *_ in rows], dtype=np.uint8)
        indices = np.array([query_indices[row[0]] for row in rows], dtype=np.int32)
        np.savez(directory / f"batch-{batch_no:06d}.npz", X=features, y=labels, q=indices)
        count += len(rows)
        if batch_no % 10 == 0:
            log(f"Training evidence: {count:,} pairs")
    manifest.write_text(json.dumps({"version": FEATURE_VERSION, "batches": batch_no + 1, "pairs": count}))


def metric_arrays(y, scores, query_index, truth_counts, mask, threshold):
    selected = scores >= threshold
    tp = np.bincount(query_index[selected], weights=y[selected], minlength=len(truth_counts))
    predicted = np.bincount(query_index[selected], minlength=len(truth_counts))
    denominator = .25 * truth_counts + predicted
    values = np.divide(1.25 * tp, denominator, out=np.zeros(len(truth_counts)), where=denominator>0)
    values[(truth_counts==0) & (predicted==0)] = 1
    return float(values[mask].mean()), values, predicted


def train(con, root, work, batch_size):
    truth = load_truth(con, root)
    build_training_features(con, work, truth, batch_size)
    con.execute("SET memory_limit='128MB'")
    directory = work / "train_features"
    queries = json.loads((directory / "queries.json").read_text())
    roles = np.array([role for _, role in queries])
    counts = np.array([len(truth[qid]) for qid, _ in queries])
    manifest = json.loads((directory / "complete.json").read_text())
    arrays = []
    for i in range(manifest["batches"]):
        with np.load(directory / f"batch-{i:06d}.npz") as part:
            arrays.append((part["X"], part["y"], part["q"]))
    X, y, qi = (np.concatenate([part[i] for part in arrays]) for i in range(3))
    del arrays
    fitting = roles[qi] == "fit"
    # Retain every positive and a reproducible negative sample within fitting IDs.
    negatives = np.flatnonzero(fitting & (y==0))
    if len(negatives)>800000:
        fitting[negatives] = False
        fitting[np.random.default_rng(20260925).choice(negatives,800000,replace=False)] = True
    tuning = roles[qi] == "tune"
    log(f"Fitting on {fitting.sum():,} pairs; tuning on {tuning.sum():,} pairs")
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=.06, num_leaves=31,
        max_depth=-1, min_child_samples=80, reg_lambda=4, colsample_bytree=.9,
        subsample=.85, subsample_freq=1, n_jobs=2, random_state=20260925, verbosity=-1)
    model.fit(X[fitting], y[fitting], eval_set=[(X[tuning],y[tuning])],
        callbacks=[lgb.early_stopping(35, verbose=False)], feature_name=FEATURE_NAMES)
    scores = model.predict_proba(X)[:,1]
    curve = []
    for threshold in np.arange(.10, .991, .01):
        value, _, _ = metric_arrays(y,scores,qi,counts,roles=="tune",float(threshold))
        curve.append({"threshold": round(float(threshold),2), "macro_f05":value})
    best = max(curve, key=lambda row:(row["macro_f05"],row["threshold"]))
    value, entity_scores, predicted = metric_arrays(y,scores,qi,counts,roles=="holdout",best["threshold"])
    oracle, oracle_scores, _ = metric_arrays(y,y.astype(float),qi,counts,roles=="holdout",.5)
    retrieved = np.bincount(qi,weights=y,minlength=len(counts))
    countries = dict(con.execute("SELECT id,country FROM queries").fetchall())
    report = {"version":VERSION,"feature_version":FEATURE_VERSION,
        "threshold":best["threshold"], "tune_macro_f05":best["macro_f05"],
        "holdout_macro_f05":value,"holdout_oracle_macro_f05":oracle,
        "iterations":model.best_iteration_, "threshold_curve":curve,"slices":{}}
    for role in ("tune","holdout"):
        for country in ("all", *sorted(set(countries.values()))):
            mask = (roles==role) & np.array([country=="all" or countries[qid]==country for qid,_ in queries])
            singletons = mask & (counts==0)
            non_singletons = mask & (counts>0)
            report["slices"][role+"/"+country] = {
                "entities":int(mask.sum()), "macro_f05":float(entity_scores[mask].mean()),
                "pair_recall":float(retrieved[mask].sum()/max(1,counts[mask].sum())),
                "complete_retrieval":float((retrieved[non_singletons]==counts[non_singletons]).mean()),
                "singleton_accuracy":float((predicted[singletons]==0).mean()) if singletons.any() else None,
                "oracle_macro_f05":float(oracle_scores[mask].mean())}
    report["features"] = dict(zip(FEATURE_NAMES, map(int,model.feature_importances_)))
    model.booster_.save_model(str(work / "model.txt"))
    fast_model = lgb.LGBMClassifier(n_estimators=100,learning_rate=.1,num_leaves=15,
        min_child_samples=80,reg_lambda=4,n_jobs=2,random_state=20260925,verbosity=-1)
    fast_x = X[:,FAST_FEATURE_INDICES]
    fast_model.fit(fast_x[fitting],y[fitting])
    fast_scores = fast_model.predict_proba(fast_x)[:,1]
    fast_threshold = 0.0
    for candidate_threshold in (.001,.002,.005,.01,.02,.03,.05,.075,.10,.15,.20):
        keep = fast_scores>=candidate_threshold
        recall = float(y[tuning & keep].sum()/max(1,y[tuning].sum()))
        gated_scores = np.where(keep,scores,0)
        gated_value,_,_ = metric_arrays(y,gated_scores,qi,counts,roles=="tune",best["threshold"])
        if recall>=.998 and gated_value>=best["macro_f05"]-.0005:
            fast_threshold = candidate_threshold
    keep = fast_scores>=fast_threshold
    gated_value,_,_ = metric_arrays(y,np.where(keep,scores,0),qi,counts,roles=="holdout",best["threshold"])
    report["fast_gate"] = {"threshold":fast_threshold,"retained_fraction":float(keep.mean()),
        "holdout_macro_f05":gated_value,
        "tune_positive_recall":float(y[tuning & keep].sum()/max(1,y[tuning].sum()))}
    fast_model.booster_.save_model(str(work / "fast_model.txt"))
    (work / "evaluation.json").write_text(json.dumps(report,indent=2))
    log(json.dumps({k:v for k,v in report.items() if k not in {"threshold_curve","features"}},indent=2))


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def external_sort_available():
    return os.name != "nt" and shutil.which("bash") and shutil.which("sort")


def run_external_sort(source, destination, temp_dir, skip_header=False, tab_keys=False):
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_q = shlex.quote(str(source))
    destination_q = shlex.quote(str(destination))
    temp_q = shlex.quote(str(temp_dir))
    reader = f"tail -n +2 {source_q}" if skip_header else f"cat {source_q}"
    keys = "-t $'\\t' -k1,1 -k2,2" if tab_keys else ""
    command = f"{reader} | LC_ALL=C sort -T {temp_q} -S 1024M {keys} > {destination_q}"
    subprocess.run(["bash", "-lc", command], check=True)


def grouped_pairs(path):
    current_qid = None
    current_ids = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qid, tid = parts[0], parts[1]
            if current_qid is None:
                current_qid = qid
            if qid != current_qid:
                yield current_qid, current_ids
                current_qid = qid
                current_ids = []
            current_ids.append(tid)
    if current_qid is not None:
        yield current_qid, current_ids


def write_submission_from_sorted(qids_path, pairs_path, output_path, output_column):
    pair_iter = grouped_pairs(pairs_path)
    current = next(pair_iter, None)
    pair_count = 0
    with qids_path.open("r", encoding="utf-8", newline="") as qids_handle, output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.writer(out_handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", output_column])
        for line in qids_handle:
            qid = line.rstrip("\n")
            while current is not None and current[0] < qid:
                raise ValueError(f"Prediction contains unknown Source-1 id {current[0]}")
            ids = []
            if current is not None and current[0] == qid:
                ids = current[1]
                pair_count += len(ids)
                current = next(pair_iter, None)
            writer.writerow([qid, ",".join(ids)])
    if current is not None:
        raise ValueError(f"Prediction contains unknown Source-1 id {current[0]}")
    return pair_count


def publish_with_external_sort(con, score_dir, output, accepted, inference_pairs):
    log("Publishing submission files with external disk sort")
    sort_tmp = score_dir / "sort_tmp"
    qids_raw = score_dir / "qids.unsorted.tsv"
    qids_sorted = score_dir / "qids.sorted.tsv"
    con.execute(f"COPY (SELECT id FROM qfull) TO {sql_path(qids_raw)} (HEADER false, DELIMITER '\t', QUOTE '')")
    run_external_sort(qids_raw, qids_sorted, sort_tmp)

    counts = {}
    for raw, filename, column in (
        (accepted, "matching_results.tsv", "matched_entity_ids"),
        (inference_pairs, "candidate_pairs.tsv", "candidate_entity_ids"),
    ):
        sorted_pairs = score_dir / (filename + ".sorted")
        temp = output / (filename + ".partial")
        run_external_sort(raw, sorted_pairs, sort_tmp, skip_header=True, tab_keys=True)
        counts[filename] = write_submission_from_sorted(qids_sorted, sorted_pairs, temp, column)
        log(f"{filename}: SHA256 {file_sha256(temp)}")
    return counts


def publish_with_duckdb(con, output, accepted, inference_pairs):
    log("Publishing submission files with DuckDB aggregation")
    con.execute(f"CREATE OR REPLACE TABLE accepted AS SELECT * FROM read_csv({sql_path(accepted)},delim='\t',header=true,columns={{'qid':'VARCHAR','tid':'VARCHAR','score':'DOUBLE'}})")
    con.execute(f"CREATE OR REPLACE TABLE inference_pairs AS SELECT * FROM read_csv({sql_path(inference_pairs)},delim='\t',header=true,columns={{'qid':'VARCHAR','tid':'VARCHAR'}})")
    counts = {
        "matching_results.tsv": con.execute("SELECT count(*) FROM accepted").fetchone()[0],
        "candidate_pairs.tsv": con.execute("SELECT count(*) FROM inference_pairs").fetchone()[0],
    }
    for filename, table, column in (("matching_results.tsv","accepted","matched_entity_ids"),("candidate_pairs.tsv","inference_pairs","candidate_entity_ids")):
        temp = output / (filename + ".partial")
        con.execute(f"""COPY (
            SELECT q.id AS source1_entity_id,coalesce(p.ids,'') AS {column}
            FROM qfull q LEFT JOIN (
                SELECT qid,string_agg(tid,',' ORDER BY tid) AS ids FROM {table} GROUP BY qid
            ) p ON p.qid=q.id ORDER BY q.id
        ) TO {sql_path(temp)} (HEADER,DELIMITER '\t',QUOTE '')""")
        log(f"{filename}: SHA256 {file_sha256(temp)}")
    return counts


def infer(con, work, model_dir, output, batch_size):
    report = json.loads((model_dir / "evaluation.json").read_text())
    if report["feature_version"] != FEATURE_VERSION:
        raise ValueError("Model feature version differs from inference")
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    fast_model = lgb.Booster(model_file=str(model_dir / "fast_model.txt"))
    fast_threshold = report["fast_gate"]["threshold"]
    threshold = report["threshold"]
    output.mkdir(parents=True, exist_ok=True)
    score_dir = work / "prediction_batches"
    score_dir.mkdir(exist_ok=True)
    # A complete prediction file is published only after every candidate is scored.
    accepted = score_dir / "accepted.tsv"
    inference_pairs = score_dir / "inference_pairs.tsv"
    processed = 0
    with accepted.open("w",encoding="utf-8",newline="") as handle, inference_pairs.open("w",encoding="utf-8",newline="") as candidate_handle:
        writer = csv.writer(handle,delimiter="\t",lineterminator="\n")
        writer.writerow(["qid","tid","score"])
        candidate_writer = csv.writer(candidate_handle,delimiter="\t",lineterminator="\n")
        candidate_writer.writerow(["qid","tid"])
        for batch_no, rows in enumerate(pair_batches(con,batch_size)):
            fast_scores = fast_model.predict(fast_features(rows),num_threads=2)
            filtered = [row for row,score in zip(rows,fast_scores) if score>=fast_threshold]
            candidate_writer.writerows((row[0],row[1]) for row in filtered)
            values = model.predict(pair_features(filtered),num_threads=2) if filtered else []
            for row,value in zip(filtered,values):
                if value >= threshold:
                    writer.writerow([row[0],row[1],f"{value:.8f}"])
            processed += len(rows)
            if batch_no % 20 == 0:
                log(f"Scored {processed:,} pairs")
    if external_sort_available():
        counts = publish_with_external_sort(con,score_dir,output,accepted,inference_pairs)
    else:
        counts = publish_with_duckdb(con,output,accepted,inference_pairs)
    total = con.execute("SELECT count(*) FROM pairs").fetchone()[0]
    if processed != total:
        raise ValueError("Inference did not score every candidate")
    for filename in ("candidate_pairs.tsv","matching_results.tsv"):
        (output / (filename+'.partial')).replace(output / filename)
    metadata = {"version":VERSION,"threshold":threshold,"retrieved_pairs":total,
        "scored_pairs":counts["candidate_pairs.tsv"],
        "matched_pairs":counts["matching_results.tsv"],
        "source1_rows":con.execute('SELECT count(*) FROM qfull').fetchone()[0],
        "model":str(model_dir.resolve())}
    (output / "run.json").write_text(json.dumps(metadata,indent=2))
    log(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","retrieve","train","predict","all-train","all-test"])
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--work",type=Path,default=ROOT/"data/features/supervised_v1")
    parser.add_argument("--model-dir",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"output/supervised_v1")
    parser.add_argument("--split",choices=["train","test"],default="train")
    parser.add_argument("--train-size",type=int,default=12000)
    parser.add_argument("--tune-size",type=int,default=3000)
    parser.add_argument("--holdout-size",type=int,default=3000)
    parser.add_argument("--batch-size",type=int,default=20000)
    parser.add_argument("--memory",default="1GB")
    args = parser.parse_args()
    split = "test" if args.command in {"all-test","predict"} else args.split
    root = find_dataset_root(args.data_root).root
    con = connect(args.work,split,args.memory)
    try:
        if args.command in {"prepare","all-train","all-test"}:
            prepare(con,root,split,args.work)
            select_queries(con,args.work,split,args.train_size,args.tune_size,args.holdout_size)
        if args.command in {"retrieve","all-train","all-test"}:
            retrieve(con,args.work,split)
        if args.command in {"train","all-train"}:
            train(con,root,args.work,args.batch_size)
        if args.command in {"predict","all-test"}:
            infer(con,args.work,args.model_dir or args.work,args.output,args.batch_size)
    finally:
        con.close()


if __name__ == "__main__":
    main()
