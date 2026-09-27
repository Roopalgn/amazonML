"""Second-generation resolution with raw-script transliteration and rare evidence.

Training, tuning and holdout IDs are disjoint. Vocabulary statistics are unlabeled;
token translations are fitted exclusively on fitting-query positive pairs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

from src.data.io import find_dataset_root
from src.matching.pair_features import fast_features
from src.matching.resolution_features import (
    VERSION, ALL_FEATURE_NAMES, clean_name, clean_address, learn_aliases, mapped, features,
)
from src.matching.supervised_pipeline import (
    log, sql_path, norm_sql, metric_arrays, file_sha256,
    publish_with_external_sort, publish_with_duckdb, external_sort_available,
)


def save_json(path, data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def connection(work, split, memory, threads):
    work.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(work / f'{split}.duckdb'))
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f'SET threads={threads}')
    con.execute('SET preserve_insertion_order=false')
    con.execute(f"SET temp_directory={sql_path(work / ('spill-'+split))}")
    return con


def table_exists(con, table):
    return bool(con.execute('SELECT 1 FROM information_schema.tables WHERE table_name=?',[table]).fetchone())


def corpus_weights(con, work, split):
    path=work/f'{split}.weights.json'
    if path.exists():
        return
    total=con.execute('SELECT count(*) FROM qfull').fetchone()[0]
    weights={field:{token:math.log1p(total/df) for token,df in con.execute(
        f'SELECT token,sum(df) FROM freq_{field} GROUP BY token').fetchall()} for field in ('nn','aa')}
    save_json(path,weights)
    if split=='train':
        save_json(work/'weights.json',weights)


def prepare(con, root, work, split, legacy, model_dir):
    marker=work / f'{split}.prepared.json'
    if marker.exists():
        corpus_weights(con,work,split)
        log('Reusing v2 normalized records')
        return
    con.create_function('clean_name',clean_name,['VARCHAR'],'VARCHAR')
    con.create_function('clean_address',clean_address,['VARCHAR'],'VARCHAR')
    for table,sources in (('qfull',[1]),('raw_targets',[2,3])):
        loaded=work/f'{split}.{table}.loaded.json'
        if table_exists(con,table) and loaded.exists():
            continue
        con.execute(f'DROP TABLE IF EXISTS {table}')
        for i,source in enumerate(sources):
            log(f'Normalizing original Unicode: {split} source {source}')
            path=root / split / f'{split}_source{source}.tsv'
            command=f'CREATE TABLE {table} AS' if i==0 else f'INSERT INTO {table}'
            con.execute(f"""{command} SELECT entity_id id,country,
                {norm_sql('business_name')} n,{norm_sql('business_address')} a,
                clean_name(coalesce(business_name,'')) nn,
                clean_address(coalesce(business_address,'')) aa
                FROM read_csv({sql_path(path)},delim='\t',header=true,all_varchar=true)""")
            con.execute('CHECKPOINT')
        save_json(loaded,{'version':VERSION,'sources':sources})
    if split=='train':
        con.execute(f'ATTACH {sql_path(legacy / "train.duckdb")} AS legacy (READ_ONLY)')
        con.execute('CREATE OR REPLACE TABLE queries AS SELECT q.*,l.role FROM qfull q JOIN legacy.queries l USING(id)')
        con.execute(f"""CREATE OR REPLACE TABLE truth_counts AS
            SELECT source1_entity_id qid, CASE WHEN coalesce(matched_entity_ids,'')=''
                THEN 0 ELSE len(string_split(matched_entity_ids,',')) END AS ntrue
            FROM read_csv({sql_path(root/'train/train_ground_truth.tsv')},delim='\t',header=true,all_varchar=true)
            SEMI JOIN queries ON id=source1_entity_id""")
        con.execute(f"""CREATE OR REPLACE TABLE truth AS
            SELECT source1_entity_id qid,unnest(string_split(matched_entity_ids,',')) tid
            FROM read_csv({sql_path(root/'train/train_ground_truth.tsv')},delim='\t',header=true,all_varchar=true)
            SEMI JOIN queries ON id=source1_entity_id WHERE coalesce(matched_entity_ids,'')!=''""")
        rows=con.execute("""SELECT q.id,q.nn,t.nn,q.aa,t.aa FROM truth g
            JOIN queries q ON q.id=g.qid JOIN raw_targets t ON t.id=g.tid WHERE q.role='fit'""").fetchall()
        aliases={
            'name':learn_aliases((r[0],r[1],r[2]) for r in rows),
            'address':learn_aliases((r[0],r[3],r[4]) for r in rows),
            'fit_query_sha256':hashlib.sha256('\n'.join(sorted({r[0] for r in rows})).encode()).hexdigest(),
        }
        save_json(work/'aliases.json',aliases)
        con.execute('DETACH legacy')
    else:
        aliases=json.loads((model_dir/'aliases.json').read_text())
        con.execute("CREATE OR REPLACE TABLE queries AS SELECT *,'test' AS role FROM qfull")
    log(f"Learned aliases: {len(aliases['name'])} name tokens, {len(aliases['address'])} address tokens")
    con.create_function('map_name',lambda v:clean_name(mapped(v,aliases['name'])),['VARCHAR'],'VARCHAR')
    con.create_function('map_address',lambda v:mapped(v,aliases['address']),['VARCHAR'],'VARCHAR')
    con.execute('CREATE OR REPLACE TABLE targets AS SELECT * EXCLUDE(nn,aa),map_name(nn) nn,map_address(aa) aa FROM raw_targets')
    for field in ('nn','aa'):
        log(f'Computing unlabeled corpus frequencies for {field}')
        con.execute(f"""CREATE OR REPLACE TABLE freq_{field} AS
            SELECT country,token,count(*) df FROM (
                SELECT country,unnest(list_distinct(string_split({field},' '))) token FROM qfull
            ) WHERE token!='' GROUP BY country,token""")
        con.execute(f'CREATE OR REPLACE TABLE exact_{field} AS SELECT country,{field},count(*) df FROM qfull GROUP BY ALL')
    corpus_weights(con,work,split)
    con.execute('CHECKPOINT')
    save_json(marker,{'version':VERSION,'split':split})


def keys_sql(table, family):
    if family in ('name','address','compact'):
        field='aa' if family=='address' else 'nn'
        expr=f"replace({field},' ','')" if family=='compact' else f"array_to_string(list_sort(list_distinct(string_split({field},' '))),' ')"
        return f'SELECT id,country,{expr} AS key FROM {table}'
    if family in ('rare_name','rare_address'):
        field='nn' if family=='rare_name' else 'aa'
        maximum=150 if family=='rare_name' else 35
        return f"""SELECT DISTINCT r.id,r.country,k.token AS key FROM (
            SELECT id,country,unnest(list_distinct(string_split({field},' '))) token FROM {table}
            ) r JOIN freq_{field} k USING(country,token)
            WHERE k.df<={maximum} AND length(k.token)>=3 AND NOT regexp_matches(k.token,'^[0-9]+$')"""
    if family=='number_name':
        return f"""SELECT id,country,unnest(flatten(list_transform(
            list_slice(regexp_extract_all(aa,'[0-9]+'),1,3), num -> list_transform(
                list_filter(string_split(nn,' '),tok -> length(tok)>=2),
                tok -> num || '|' || tok)))) AS key FROM {table}"""
    if family=='number_address':
        return f"""SELECT r.id,r.country,item.num || '|' || item.token AS key FROM (
            SELECT id,country,unnest(flatten(list_transform(
                list_slice(regexp_extract_all(aa,'[0-9]+'),1,3), num -> list_transform(
                    list_filter(string_split(aa,' '),tok -> length(tok)>=3 AND NOT regexp_matches(tok,'[0-9]')),
                    tok -> struct_pack(num:=num,token:=tok))))) item FROM {table}
            ) r JOIN freq_aa f ON f.country=r.country AND f.token=item.token
            WHERE f.df<=1500"""
    if family=='name_pair':
        return f"""SELECT id,country,unnest(flatten(list_transform(nt,x -> list_transform(
            list_filter(nt,y -> x<y),y -> x||'|'||y)))) AS key FROM (
            SELECT id,country,list_filter(list_distinct(string_split(nn,' ')),x -> length(x)>=2) nt FROM {table})"""
    raise ValueError(family)


def retrieve(con,work,split,legacy):
    marker=work/f'{split}.retrieved.json'
    if marker.exists():
        return
    # Legacy retrieval remains part of the union; new routes can only add candidates.
    tables=[]
    use_legacy=os.environ.get('RESOLUTION_SKIP_LEGACY','').lower() not in ('1','true','yes')
    if use_legacy and (legacy/f'{split}.duckdb').exists():
        con.execute(f'ATTACH {sql_path(legacy/f"{split}.duckdb")} AS legacy (READ_ONLY)')
        if not table_exists(con,'pairs_legacy'):
            log('Importing baseline candidate pairs')
            con.execute('CREATE TABLE pairs_legacy AS SELECT p.* FROM legacy.pairs p SEMI JOIN queries q ON q.id=p.qid')
        con.execute('DETACH legacy')
        tables.append('pairs_legacy')
    route_env=os.environ.get('RESOLUTION_ROUTE_FAMILIES')
    families=tuple(route_env.split(',')) if route_env else (
        'name','address','compact','rare_name','rare_address','number_name','number_address','name_pair'
    )
    for family in families:
        family=family.strip()
        if not family:
            continue
        table='pairs_'+family
        tables.append(table)
        if table_exists(con,table):
            continue
        log('Retrieving '+family)
        con.execute(f'CREATE OR REPLACE TABLE qkeys AS SELECT DISTINCT * FROM ({keys_sql("queries",family)})')
        con.execute("DELETE FROM qkeys WHERE key IS NULL OR length(key)<3")
        con.execute(f"""CREATE OR REPLACE TABLE tkeys AS SELECT DISTINCT t.* FROM ({keys_sql('targets',family)}) t
            SEMI JOIN qkeys q USING(country,key)""")
        limit=1500 if family in ('name','address','compact') else 400
        con.execute(f"""CREATE TABLE {table} AS WITH eligible AS (
            SELECT country,key FROM tkeys GROUP BY country,key HAVING count(*)<={limit}
        ) SELECT DISTINCT q.id qid,t.id tid FROM tkeys t JOIN eligible USING(country,key)
            JOIN qkeys q USING(country,key)""")
        log(f'{family}: {con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:,} pairs')
        con.execute('DROP TABLE qkeys')
        con.execute('DROP TABLE tkeys')
        con.execute('CHECKPOINT')
    union=' UNION ALL '.join(f'SELECT * FROM {t}' for t in tables)
    con.execute(f'CREATE OR REPLACE TABLE pairs AS SELECT DISTINCT qid,tid FROM ({union})')
    total=con.execute('SELECT count(*) FROM pairs').fetchone()[0]
    save_json(marker,{'version':VERSION,'pairs':total,'routes':tables})
    log(f'Total candidates: {total:,}')


def pair_batches(con,batch_size):
    cursor=con.execute('''SELECT p.qid,p.tid,q.n,q.a,t.n,t.a,q.nn,q.aa,t.nn,t.aa,nf.df,af.df
        FROM pairs p JOIN queries q ON q.id=p.qid JOIN targets t ON t.id=p.tid
        JOIN exact_nn nf ON nf.country=q.country AND nf.nn=q.nn
        JOIN exact_aa af ON af.country=q.country AND af.aa=q.aa''')
    while rows:=cursor.fetchmany(batch_size):
        yield rows


def gate(rows,fast_model,threshold):
    from rapidfuzz import fuzz,process
    quick=fast_features(rows)
    old=fast_model.predict(quick,num_threads=2)>=threshold
    names=process.cpdist([r[6] for r in rows],[r[8] for r in rows],scorer=fuzz.ratio,workers=2)/100
    addresses=process.cpdist([r[7] for r in rows],[r[9] for r in rows],scorer=fuzz.ratio,workers=2)/100
    return old | ((names>=.60)&(names>=quick[:,0]+.10)) | ((addresses>=.55)&(addresses>=quick[:,2]+.10)) | (addresses>=.90)


def export_training(con,work):
    directory=work/'export'
    directory.mkdir(exist_ok=True)
    queries=con.execute('SELECT q.id,q.role,q.country,c.ntrue FROM queries q JOIN truth_counts c ON c.qid=q.id ORDER BY q.id').fetchall()
    save_json(directory/'queries.json',queries)
    con.execute('CREATE OR REPLACE TEMP TABLE qindex AS SELECT id,row_number() OVER(ORDER BY id)-1 qi FROM queries')
    log('Exporting portable training evidence')
    con.execute(f"""COPY (SELECT p.qid,p.tid,q.n,q.a,t.n tn,t.a ta,q.nn,q.aa,t.nn tnn,t.aa taa,
        nf.df name_frequency,af.df address_frequency,i.qi,(g.qid IS NOT NULL)::INTEGER y
        FROM pairs p JOIN queries q ON q.id=p.qid JOIN targets t ON t.id=p.tid
        JOIN exact_nn nf ON nf.country=q.country AND nf.nn=q.nn
        JOIN exact_aa af ON af.country=q.country AND af.aa=q.aa
        JOIN qindex i ON i.id=q.id LEFT JOIN truth g USING(qid,tid)
        ) TO {sql_path(directory/'pairs.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 50000)""")
    oracle=con.execute('''WITH retrieved AS (
        SELECT p.qid,count(*) n FROM pairs p JOIN truth USING(qid,tid) GROUP BY p.qid)
        SELECT q.role,q.country,count(*) entities,sum(c.ntrue) positives,
            sum(coalesce(r.n,0)) retrieved_positives,
            avg(CASE WHEN c.ntrue=0 THEN 1. ELSE 1.25*coalesce(r.n,0)/(.25*c.ntrue+coalesce(r.n,0)) END) oracle
        FROM queries q JOIN truth_counts c ON q.id=c.qid LEFT JOIN retrieved r ON r.qid=q.id GROUP BY ALL''').fetchall()
    save_json(directory/'retrieval_audit.json',{'columns':['role','country','entities','positives','retrieved','oracle'],'rows':oracle})
    log(json.dumps({'export_bytes':(directory/'pairs.parquet').stat().st_size,'retrieval_oracle':oracle}))


def build_features(con,work,model_dir,legacy,batch_size):
    directory=work/'train_features'
    directory.mkdir(exist_ok=True)
    marker=directory/'complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['version']!=VERSION:
            raise ValueError('Feature cache version mismatch; use a fresh work directory')
        return
    for unfinished in directory.glob('batch-*.npz'):
        unfinished.unlink()
    queries=con.execute('SELECT q.id,q.role,q.country,c.ntrue FROM queries q JOIN truth_counts c ON c.qid=q.id ORDER BY q.id').fetchall()
    qindices={q[0]:i for i,q in enumerate(queries)}
    truth=set(con.execute('SELECT qid,tid FROM truth').fetchall())
    save_json(directory/'queries.json',queries)
    weights=json.loads((model_dir/'weights.json').read_text())
    fast_model=lgb.Booster(model_file=str(legacy/'fast_model.txt'))
    threshold=json.loads((legacy/'evaluation.json').read_text())['fast_gate']['threshold']
    total=retained=positive_retrieved=positive_kept=0
    for batch,rows in enumerate(pair_batches(con,batch_size)):
        keep=gate(rows,fast_model,threshold)
        labels=np.array([(r[0],r[1]) in truth for r in rows],dtype=np.uint8)
        positive_retrieved+=int(labels.sum())
        positive_kept+=int(labels[keep].sum())
        kept=[r for r,k in zip(rows,keep) if k]
        if kept:
            X=features(kept,weights['nn'],weights['aa'])
            np.savez(directory/f'batch-{batch:06d}.npz',X=X,y=labels[keep],
                q=np.array([qindices[r[0]] for r in kept],dtype=np.int32),
                tid=np.array([r[1] for r in kept]))
        total+=len(rows)
        retained+=len(kept)
        if batch%10==0:
            log(f'Features: {total:,} retrieved / {retained:,} retained')
    save_json(marker,{'version':VERSION,'retrieved':total,'retained':retained,
                     'positive_retrieved':positive_retrieved,'positive_kept':positive_kept})


def evaluate(y,scores,qi,counts,roles,countries,threshold):
    report={}
    for role in ('tune','holdout'):
        for country in ('all',*sorted(set(countries))):
            mask=(roles==role)&((countries==country) if country!='all' else True)
            value,values,predicted=metric_arrays(y,scores,qi,counts,mask,threshold)
            oracle,_,_=metric_arrays(y,y,qi,counts,mask,.5)
            single=mask&(counts==0)
            report[role+'/'+country]={'entities':int(mask.sum()),'macro_f05':value,'oracle_macro_f05':oracle,
                'singleton_accuracy':float((predicted[single]==0).mean()) if single.any() else None,
                'perfect_entities':int((values[mask]==1).sum())}
    return report


def train(con,work,legacy,batch_size,threads):
    build_features(con,work,work,legacy,batch_size)
    con.execute("SET memory_limit='128MB'")
    directory=work/'train_features'
    queries=json.loads((directory/'queries.json').read_text())
    roles=np.array([r[1] for r in queries])
    countries=np.array([r[2] for r in queries])
    counts=np.array([r[3] for r in queries])
    meta=json.loads((directory/'complete.json').read_text())
    X=np.lib.format.open_memmap(directory/'X.npy',mode='w+',dtype=np.float32,shape=(meta['retained'],len(ALL_FEATURE_NAMES)))
    y=np.empty(meta['retained'],dtype=np.uint8)
    qi=np.empty(meta['retained'],dtype=np.int32)
    offset=0
    for file in sorted(directory.glob('batch-*.npz')):
        with np.load(file) as part:
            size=len(part['y'])
            X[offset:offset+size]=part['X']
            y[offset:offset+size]=part['y']
            qi[offset:offset+size]=part['q']
            offset+=size
    assert offset==len(y)
    fit=roles[qi]=='fit'
    tune=roles[qi]=='tune'
    # Macro scoring gives each query equal importance, including singleton queries.
    sample_weight=1/np.maximum(1,counts[qi[fit]])**.5
    model=lgb.LGBMClassifier(n_estimators=1400,learning_rate=.045,num_leaves=63,
        min_child_samples=35,reg_lambda=3,colsample_bytree=.9,
        subsample=.9,subsample_freq=1,n_jobs=threads,random_state=20260926,verbosity=-1,
        force_col_wise=True)
    log(f'Fitting v2 on {int(fit.sum()):,} pairs')
    model.fit(X[fit],y[fit],sample_weight=sample_weight,eval_set=[(X[tune],y[tune])],
        callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(100)],feature_name=ALL_FEATURE_NAMES)
    scores=np.empty(len(y),dtype=np.float32)
    for start in range(0,len(y),50000):
        scores[start:start+50000]=model.booster_.predict(X[start:start+50000],num_threads=threads)
    curve=[]
    for threshold in np.arange(.10,.991,.01):
        value,_,_=metric_arrays(y,scores,qi,counts,roles=='tune',threshold)
        curve.append({'threshold':round(float(threshold),2),'macro_f05':value})
    best=max(curve,key=lambda r:(r['macro_f05'],r['threshold']))
    slices=evaluate(y,scores,qi,counts,roles,countries,best['threshold'])
    report={'version':VERSION,'threshold':best['threshold'],'tune_macro_f05':best['macro_f05'],
        'holdout_macro_f05':slices['holdout/all']['macro_f05'],'slices':slices,'curve':curve,
        'iterations':model.best_iteration_,'features':dict(zip(ALL_FEATURE_NAMES,map(int,model.feature_importances_))),
        'gate':meta,'validation_note':'Legacy holdout is a development comparison, not a fresh final test.'}
    model.booster_.save_model(str(work/'model.txt'))
    np.savez(directory/'predictions.npz',y=y,q=qi,score=scores)
    save_json(work/'evaluation.json',report)
    log(json.dumps({k:v for k,v in report.items() if k not in ('features','curve')}))


def predict(con,work,model_dir,legacy,output,batch_size,threads):
    report=json.loads((model_dir/'evaluation.json').read_text())
    if report['version']!=VERSION:
        raise ValueError('Model version mismatch')
    # Unlabeled test frequencies prevent common unseen French words being treated
    # as rare simply because they did not appear in the English/Indian training set.
    corpus_weights(con,work,'test')
    weights=json.loads((work/'test.weights.json').read_text())
    model=lgb.Booster(model_file=str(model_dir/'model.txt'))
    fast_model=lgb.Booster(model_file=str(legacy/'fast_model.txt'))
    fast_threshold=json.loads((legacy/'evaluation.json').read_text())['fast_gate']['threshold']
    output.mkdir(parents=True,exist_ok=True)
    scores_dir=work/'prediction_batches'
    scores_dir.mkdir(exist_ok=True)
    accepted=scores_dir/'accepted.tsv'
    candidates=scores_dir/'inference_pairs.tsv'
    # Persist scores for rapid threshold experiments without repeating inference.
    all_scores=scores_dir/'scores.tsv'
    processed=0
    with accepted.open('w',encoding='utf-8',newline='') as ah,candidates.open('w',encoding='utf-8',newline='') as ch,all_scores.open('w',encoding='utf-8',newline='') as sh:
        aw,cw,sw=[csv.writer(h,delimiter='\t',lineterminator='\n') for h in (ah,ch,sh)]
        aw.writerow(['qid','tid','score'])
        cw.writerow(['qid','tid'])
        sw.writerow(['qid','tid','score'])
        for batch,rows in enumerate(pair_batches(con,batch_size)):
            keep=gate(rows,fast_model,fast_threshold)
            filtered=[r for r,k in zip(rows,keep) if k]
            if filtered:
                probabilities=model.predict(features(filtered,weights['nn'],weights['aa']),num_threads=threads)
                cw.writerows((r[0],r[1]) for r in filtered)
                sw.writerows((r[0],r[1],f'{p:.8f}') for r,p in zip(filtered,probabilities) if p>=.01)
                aw.writerows((r[0],r[1],f'{p:.8f}') for r,p in zip(filtered,probabilities) if p>=report['threshold'])
            processed+=len(rows)
            if batch%20==0:
                log(f'Predicted {processed:,} pairs')
    if external_sort_available():
        counts=publish_with_external_sort(con,scores_dir,output,accepted,candidates)
    else:
        counts=publish_with_duckdb(con,output,accepted,candidates)
    if processed!=con.execute('SELECT count(*) FROM pairs').fetchone()[0]:
        raise ValueError('Incomplete inference')
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        (output/(name+'.partial')).replace(output/name)
    save_json(output/'run.json',{'version':VERSION,'threshold':report['threshold'],'retrieved':processed,
        'counts':counts,'matching_sha256':file_sha256(output/'matching_results.tsv')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','retrieve','train','predict','all-train','all-test','export'])
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--work',type=Path,default=Path('data/features/resolution_v2'))
    parser.add_argument('--legacy',type=Path,default=Path('data/features/supervised_v1'))
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('output/resolution_v2'))
    parser.add_argument('--split',choices=['train','test'],default='train')
    parser.add_argument('--memory',default='768MB')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=10000)
    args=parser.parse_args()
    split='test' if args.command in ('all-test','predict') else args.split
    model_dir=args.model_dir or args.work
    con=connection(args.work,split,args.memory,args.threads)
    try:
        if args.command in ('prepare','all-train','all-test'):
            prepare(con,find_dataset_root(args.data_root).root,args.work,split,args.legacy,model_dir)
        if args.command in ('retrieve','all-train','all-test'):
            retrieve(con,args.work,split,args.legacy)
        if args.command in ('train','all-train'):
            train(con,args.work,args.legacy,args.batch_size,args.threads)
        if args.command=='export':
            export_training(con,args.work)
        if args.command in ('predict','all-test'):
            predict(con,args.work,model_dir,args.legacy,args.output,args.batch_size,args.threads)
    finally:
        con.close()


if __name__=='__main__':
    main()
