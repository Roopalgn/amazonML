"""Resolve competing owners of a target; validate against the full query corpus.

The supplied training graph has one Source-1 owner per target. Evaluation still
uses the official multi-match macro metric: one query may own many targets.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

from src.data.io import find_dataset_root
from src.eval.macro_f05 import entity_f05
from src.matching.pair_features import fast_features,pair_features
from src.matching.supervised_pipeline import key_expression,log,sql_path,file_sha256


def confidence_gap(best,runner_up):
    def odds(value):
        value=min(1-1e-9,max(1e-9,value))
        return math.log(value/(1-value))
    return odds(best)-odds(runner_up)


def score_pairs(con,query,model_dir,destination,gate=True):
    model=lgb.Booster(model_file=str(model_dir/'model.txt'))
    fast=lgb.Booster(model_file=str(model_dir/'fast_model.txt'))
    report=json.loads((model_dir/'evaluation.json').read_text())
    total=0
    partial=destination.with_suffix('.partial')
    with partial.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n')
        writer.writerow(['qid','tid','score'])
        cursor=con.execute(query)
        while rows:=cursor.fetchmany(10000):
            if gate:
                values=fast.predict(fast_features(rows),num_threads=2)
                selected=[r for r,p in zip(rows,values) if p>=report['fast_gate']['threshold']]
            else:
                selected=rows
            if selected:
                probabilities=model.predict(pair_features(selected),num_threads=2)
                writer.writerows((r[0],r[1],f'{p:.12f}') for r,p in zip(selected,probabilities) if not gate or p>=report['threshold'])
            total+=len(rows)
            if total%100000==0:
                log(f'Competition scoring: {total:,} pairs')
    partial.replace(destination)
    destination.with_suffix('.complete.json').write_text(json.dumps({'scored_pairs':total}))
    return total


def ranked_table(con,source,table='ranked'):
    con.execute(f"""CREATE OR REPLACE TABLE {table} AS SELECT *,
        row_number() OVER(PARTITION BY tid ORDER BY score DESC,qid) rank,
        lead(score) OVER(PARTITION BY tid ORDER BY score DESC,qid) runner,
        count(*) OVER(PARTITION BY tid) owners
        FROM {source}""")


def selected_owners(rows,margin):
    result={}
    for qid,tid,score,rank,runner,owners in rows:
        if rank==1 and (owners==1 or (score>runner and confidence_gap(score,runner)>=margin)):
            result[tid]=qid
    return result


def validate(con,legacy,work):
    con.execute(f'ATTACH {sql_path(legacy/"train.duckdb")} AS legacy (READ_ONLY)')
    original=work/'validation_original.tsv'
    if not original.exists() or not original.with_suffix('.complete.json').exists():
        log('Scoring validation queries with baseline')
        score_pairs(con,"""SELECT p.qid,p.tid,q.n,q.a,t.n,t.a FROM legacy.pairs p
            JOIN legacy.queries q ON q.id=p.qid JOIN legacy.targets t ON t.id=p.tid
            WHERE q.role IN ('tune','holdout')""",legacy,original)
    con.execute(f"CREATE OR REPLACE TABLE initial AS SELECT * FROM read_csv({sql_path(original)},delim='\t',header=true)")
    con.execute('CREATE OR REPLACE TABLE selected_targets AS SELECT * FROM legacy.targets SEMI JOIN initial ON id=tid')
    competitor_scores=work/'validation_competitors.tsv'
    if not competitor_scores.exists() or not competitor_scores.with_suffix('.complete.json').exists():
        for family in ('name','address','name_pair','house_name','house_locality','locality_pair'):
            log('Full-corpus competitors: '+family)
            expr=key_expression(family)
            con.execute(f'CREATE OR REPLACE TABLE tkeys AS SELECT DISTINCT id tid,country,unnest({expr}) AS key FROM selected_targets')
            con.execute("DELETE FROM tkeys WHERE key IS NULL OR length(key)<4")
            con.execute(f"""CREATE OR REPLACE TABLE qkeys AS SELECT q.* FROM (
                SELECT DISTINCT id qid,country,unnest({expr}) AS key FROM legacy.qfull) q
                SEMI JOIN tkeys USING(country,key)""")
            con.execute(f"""CREATE OR REPLACE TABLE competitors_{family} AS
                WITH eligible AS (SELECT country,key FROM qkeys GROUP BY country,key HAVING count(*)<=1500)
                SELECT DISTINCT qid,tid FROM qkeys JOIN eligible USING(country,key) JOIN tkeys USING(country,key)""")
        union=' UNION ALL '.join(f'SELECT * FROM competitors_{f}' for f in ('name','address','name_pair','house_name','house_locality','locality_pair'))
        con.execute(f'CREATE OR REPLACE TABLE competitors AS SELECT DISTINCT qid,tid FROM ({union} UNION ALL SELECT qid,tid FROM initial)')
        log('Scoring competitors across every Source 1 record')
        score_pairs(con,"""SELECT p.qid,p.tid,q.n,q.a,t.n,t.a FROM competitors p
            JOIN legacy.qfull q ON q.id=p.qid JOIN selected_targets t ON t.id=p.tid""",legacy,competitor_scores)
    con.execute(f"CREATE OR REPLACE TABLE scores AS SELECT * FROM read_csv({sql_path(competitor_scores)},delim='\t',header=true)")
    ranked_table(con,'scores')
    ranked=con.execute('SELECT * FROM ranked WHERE rank=1').fetchall()
    root=find_dataset_root().root
    truth_rows=con.execute(f"""SELECT source1_entity_id,matched_entity_ids,q.role,q.country
        FROM read_csv({sql_path(root/'train/train_ground_truth.tsv')},delim='\t',header=true,all_varchar=true)
        JOIN legacy.queries q ON q.id=source1_entity_id WHERE q.role IN ('tune','holdout')""").fetchall()
    baseline={qid:set() for qid,*_ in truth_rows}
    for qid,tid in con.execute('SELECT qid,tid FROM initial').fetchall():
        baseline[qid].add(tid)
    truth={qid:set(ids.split(',')) if ids else set() for qid,ids,*_ in truth_rows}
    curve=[]
    for margin in (0.,.1,.25,.5,1.,2.,3.):
        owner=selected_owners(ranked,margin)
        predictions={q:{t for t in ts if owner.get(t)==q} for q,ts in baseline.items()}
        value=np.mean([entity_f05(truth[q],predictions[q]) for q,_,role,_ in truth_rows if role=='tune'])
        curve.append({'margin':margin,'tune_macro_f05':float(value)})
    best=max(curve,key=lambda r:(r['tune_macro_f05'],-r['margin']))
    owner=selected_owners(ranked,best['margin'])
    predictions={q:{t for t in ts if owner.get(t)==q} for q,ts in baseline.items()}
    report={'version':'ownership-v1','margin':best['margin'],'curve':curve,'slices':{}}
    for role in ('tune','holdout'):
        for country in ('all','India','US'):
            ids=[q for q,_,r,c in truth_rows if r==role and (country=='all' or c==country)]
            report['slices'][role+'/'+country]={
                'entities':len(ids),'baseline':float(np.mean([entity_f05(truth[q],baseline[q]) for q in ids])),
                'resolved':float(np.mean([entity_f05(truth[q],predictions[q]) for q in ids])),
                'removed_true':sum(len((baseline[q]-predictions[q])&truth[q]) for q in ids),
                'removed_false':sum(len((baseline[q]-predictions[q])-truth[q]) for q in ids),
            }
    (work/'evaluation.json').write_text(json.dumps(report,indent=2))
    log(json.dumps(report,indent=2))


def predict(con,legacy,work,source,output,evaluation):
    report=json.loads(evaluation.read_text())
    if report['slices']['tune/all']['resolved']<=report['slices']['tune/all']['baseline']:
        raise ValueError('Ownership resolver did not improve tuning score')
    con.execute(f'ATTACH {sql_path(legacy/"test.duckdb")} AS legacy (READ_ONLY)')
    con.execute(f"""CREATE OR REPLACE TABLE links AS SELECT source1_entity_id qid,
        unnest(string_split(matched_entity_ids,',')) tid FROM read_csv({sql_path(source)},
        delim='\t',header=true,all_varchar=true) WHERE coalesce(matched_entity_ids,'')!=''""")
    con.execute('CREATE OR REPLACE TABLE conflicts AS SELECT tid FROM links GROUP BY tid HAVING count(*)>1')
    conflict_slices=con.execute('''SELECT q.country,count(*) links,count(DISTINCT p.qid) queries,
        count(DISTINCT p.tid) targets FROM links p SEMI JOIN conflicts USING(tid)
        JOIN legacy.qfull q ON q.id=p.qid GROUP BY q.country''').fetchall()
    log('Test conflicts by country: '+json.dumps(conflict_slices))
    scored=work/'test_conflicts.tsv'
    if not scored.exists() or not scored.with_suffix('.complete.json').exists():
        score_pairs(con,"""SELECT p.qid,p.tid,q.n,q.a,t.n,t.a FROM links p
            SEMI JOIN conflicts USING(tid) JOIN legacy.qfull q ON q.id=p.qid
            JOIN legacy.targets t ON t.id=p.tid""",legacy,scored,gate=False)
    con.execute(f"CREATE OR REPLACE TABLE scores AS SELECT * FROM read_csv({sql_path(scored)},delim='\t',header=true)")
    ranked_table(con,'scores')
    owners=selected_owners(con.execute('SELECT * FROM ranked WHERE rank=1').fetchall(),report['margin'])
    conflicted={r[0] for r in con.execute('SELECT tid FROM conflicts').fetchall()}
    output.mkdir(parents=True,exist_ok=True)
    destination=output/'matching_results.tsv'
    rows=removed=0
    with source.open(encoding='utf-8',newline='') as inp,destination.with_suffix('.partial').open('w',encoding='utf-8',newline='') as out:
        reader=csv.DictReader(inp,delimiter='\t')
        writer=csv.writer(out,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id','matched_entity_ids'])
        for row in reader:
            qid=row['source1_entity_id']
            ids=row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            selected=[tid for tid in ids if tid not in conflicted or owners.get(tid)==qid]
            removed+=len(ids)-len(selected)
            writer.writerow([qid,','.join(selected)])
            rows+=1
    destination.with_suffix('.partial').replace(destination)
    metadata={'version':'ownership-v1','source_sha256':file_sha256(source),
        'matching_sha256':file_sha256(destination),'rows':rows,'removed_links':removed,
        'conflicted_targets':len(conflicted),'margin':report['margin'],'country_conflicts':conflict_slices,
        'candidate_pairs':'Use the original ec2_supervised/candidate_pairs.tsv; this step only removes links.'}
    (output/'run.json').write_text(json.dumps(metadata,indent=2))
    log(json.dumps(metadata))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['validate','predict'])
    parser.add_argument('--legacy',type=Path,default=Path('data/features/supervised_v1'))
    parser.add_argument('--work',type=Path,default=Path('data/features/ownership_v1'))
    parser.add_argument('--input',type=Path,default=Path('output/ec2_supervised/matching_results.tsv'))
    parser.add_argument('--output',type=Path,default=Path('output/ownership_v1'))
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(args.work/f'{args.command}.duckdb'))
    con.execute("SET memory_limit='512MB'")
    con.execute('SET threads=2')
    con.execute('SET enable_progress_bar=false')
    con.execute(f"SET temp_directory={sql_path(args.work/'spill')}")
    try:
        if args.command=='validate':
            validate(con,args.legacy,args.work)
        else:
            predict(con,args.legacy,args.work,args.input,args.output,args.work/'evaluation.json')
    finally:
        con.close()


if __name__=='__main__':
    main()
