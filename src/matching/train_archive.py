"""Train v2 from a portable evidence parquet, preserving the original split."""

import argparse
import json
from pathlib import Path
import shutil

import duckdb
import lightgbm as lgb
import numpy as np

from src.matching.resolution_features import VERSION,features
from src.matching.resolution_pipeline import gate,train,save_json
from src.matching.supervised_pipeline import log,sql_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    directory=args.work/'train_features'
    directory.mkdir(exist_ok=True)
    for filename in ('aliases.json','weights.json'):
        shutil.copy2(args.archive/filename,args.work/filename)
    shutil.copy2(args.archive/'queries.json',directory/'queries.json')
    con=duckdb.connect()
    con.execute("SET memory_limit='512MB'")
    con.execute(f'SET threads={args.threads}')
    if not (directory/'complete.json').exists():
        weights=json.loads((args.work/'weights.json').read_text())
        fast=lgb.Booster(model_file=str(args.legacy/'fast_model.txt'))
        threshold=json.loads((args.legacy/'evaluation.json').read_text())['fast_gate']['threshold']
        cursor=con.execute(f'SELECT * FROM read_parquet({sql_path(args.archive/"pairs.parquet")})')
        total=retained=positive_retrieved=positive_kept=0
        batch=0
        while rows:=cursor.fetchmany(10000):
            keep=gate(rows,fast,threshold)
            labels=np.array([r[13] for r in rows],dtype=np.uint8)
            kept=[r for r,k in zip(rows,keep) if k]
            if kept:
                X=features(kept,weights['nn'],weights['aa'])
                np.savez(directory/f'batch-{batch:06d}.npz',X=X,y=labels[keep],
                    q=np.array([r[12] for r in kept],dtype=np.int32),tid=np.array([r[1] for r in kept]))
            total+=len(rows)
            retained+=len(kept)
            positive_retrieved+=int(labels.sum())
            positive_kept+=int(labels[keep].sum())
            if batch%20==0:
                log(f'Archive features: {total:,} retrieved / {retained:,} retained')
            batch+=1
        save_json(directory/'complete.json',{'version':VERSION,'retrieved':total,'retained':retained,
            'positive_retrieved':positive_retrieved,'positive_kept':positive_kept})
        del weights,fast
    train(con,args.work,args.legacy,10000,args.threads)
    con.close()


if __name__=='__main__':
    main()
