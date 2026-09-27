"""Audit retrieval and classification separately using only supplied labels."""

import argparse
import csv
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

from src.data.io import find_dataset_root
from src.matching.pair_features import pair_features
from src.matching.supervised_pipeline import log, sql_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('data/features/supervised_v1'))
    parser.add_argument('--output', type=Path, default=Path('output/audit_v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.work / 'train.duckdb'), read_only=True)
    con.execute("SET memory_limit='512MB'")
    con.execute('SET threads=2')
    root = find_dataset_root().root
    con.execute(f"""CREATE TEMP TABLE truth AS
        SELECT source1_entity_id qid, unnest(string_split(matched_entity_ids, ',')) tid
        FROM read_csv({sql_path(root / 'train/train_ground_truth.tsv')},
            delim='\t', header=true, all_varchar=true)
        SEMI JOIN queries ON source1_entity_id=id WHERE matched_entity_ids!=''""")
    model = lgb.Booster(model_file=str(args.work / 'model.txt'))
    rows = con.execute("""SELECT g.qid,g.tid,q.n,q.a,t.n,t.a,q.country,q.role,
            p.qid IS NOT NULL retrieved
        FROM truth g JOIN queries q ON q.id=g.qid JOIN targets t ON t.id=g.tid
        LEFT JOIN pairs p USING(qid,tid) ORDER BY g.qid,g.tid""").fetchall()
    scores = np.concatenate([model.predict(pair_features(rows[i:i+10000]), num_threads=2)
                             for i in range(0, len(rows), 10000)])
    columns = ['qid','tid','name','address','target_name','target_address','country','role','retrieved','score']
    with (args.output / 'positive_pairs.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(columns)
        writer.writerows([*row, float(score)] for row, score in zip(rows, scores))
    summary = {}
    for country in ('all', 'India', 'US'):
        selected = [(r, float(s)) for r,s in zip(rows,scores) if r[7]!='fit' and (country=='all' or r[6]==country)]
        summary[country] = {
            'positive_pairs': len(selected),
            'retrieval_misses': sum(not r[8] for r,s in selected),
            'classifier_misses_at_091': sum(s<.91 for r,s in selected),
            'retrieved_classifier_misses_at_091': sum(r[8] and s<.91 for r,s in selected),
        }
    (args.output / 'summary.json').write_text(json.dumps(summary,indent=2))
    log(json.dumps(summary))
    for row, score in [(r,s) for r,s in zip(rows,scores) if r[7]=='tune' and not r[8]][:35]:
        print(json.dumps({'type':'retrieval_miss','row':row,'score':float(score)},ensure_ascii=True))
    for row, score in sorted([(r,s) for r,s in zip(rows,scores) if r[7]=='tune' and r[8]],key=lambda x:x[1])[:25]:
        print(json.dumps({'type':'classification_miss','row':row,'score':float(score)},ensure_ascii=True))
    con.close()


if __name__ == '__main__':
    main()
