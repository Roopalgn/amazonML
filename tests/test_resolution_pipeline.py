import unittest
import csv
import json
from pathlib import Path
import tempfile

import duckdb
import numpy as np

from src.matching.resolution_features import (
    ALL_FEATURE_NAMES, clean_address, clean_name, features, latin, learn_aliases,
    fuzzy_batch, fuzzy_overlap,
)
from src.matching.resolution_pipeline import keys_sql,prepare


class ResolutionTests(unittest.TestCase):
    def test_transliteration_preserves_indic_vowels(self):
        raw='\u092c\u094d\u0932\u0942 \u092c\u093f\u091c\u0928\u0947\u0938'
        self.assertEqual(latin(raw),'blu bijnes')
        self.assertEqual(latin('caf\u00e9'),'cafe')

    def test_number_and_suffix_normalization(self):
        self.assertEqual(clean_address('No. 00128 25th Street'),'no 128 25 st')
        self.assertEqual(clean_name('Acme Private Limited ID 12345'),'acme')

    def test_aliases_require_multiple_entities_and_consistent_evidence(self):
        repeated=[('q1','blue business','blu bijnes')]*10
        self.assertEqual(learn_aliases(repeated),{})
        rows=[(f'q{i}',f'blue shop{i}',f'blu shop{i}') for i in range(5)]
        self.assertEqual(learn_aliases(rows),{'blu':'blue'})

    def test_features_are_finite_and_rarity_distinguishes_overlap(self):
        rows=[('q','t','acme ltd','128 main street','acme ltd','128 main street',
               'acme','128 main st','acme','128 main st',1,1),
              ('q2','t2','','','','','','','','',1,1)]
        matrix=features(rows,{'acme':8},{'128':4,'main':2,'st':1})
        self.assertEqual(matrix.shape,(2,len(ALL_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix[0,ALL_FEATURE_NAMES.index('name_idf_recall')],1)
        self.assertEqual(matrix[1,ALL_FEATURE_NAMES.index('name_idf_recall')],0)

    def test_batched_fuzzy_features_match_individual_reference(self):
        left=['blue blue business','128 main road','','acme','blue business']
        right=['blu business','128 maim rd','acme','','business blue']
        weights={'blue':7,'business':3,'128':5,'main':4,'road':1}
        expected=np.array([fuzzy_overlap(list(dict.fromkeys(a.split())),list(dict.fromkeys(b.split())),weights)
                           for a,b in zip(left,right)])
        np.testing.assert_allclose(fuzzy_batch(left,right,weights,chunk_size=2),expected,rtol=1e-6,atol=1e-6)

    def test_retrieval_keys_cover_short_name_and_nonleading_house(self):
        con=duckdb.connect()
        try:
            con.execute("CREATE TABLE records AS SELECT 'q' id,'India' country,'vvf engineers' nn,'no 494 128 main st' aa")
            rows=con.execute(keys_sql('records','number_name')).fetchall()
            self.assertIn(('q','India','128|vvf'),rows)
            con.execute("CREATE TABLE freq_nn AS SELECT 'India' country,'vvf' token,5 df")
            con.execute("CREATE TABLE freq_aa AS SELECT 'India' country,'main' token,5 df")
            for route in ('name','address','compact','rare_name','rare_address','number_address','name_pair'):
                self.assertTrue(con.execute(keys_sql('records',route)).fetchall(),route)
        finally:
            con.close()

    def test_preparation_never_learns_translations_from_tuning_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'train').mkdir()
            legacy=root/'legacy'
            legacy.mkdir()
            con=duckdb.connect(str(legacy/'train.duckdb'))
            con.execute('CREATE TABLE queries(id VARCHAR,role VARCHAR)')
            con.executemany('INSERT INTO queries VALUES (?,?)',[(f'q{i}','fit' if i<4 else 'tune') for i in range(8)])
            con.close()
            sources={1:[],2:[],3:[]}
            for i in range(8):
                sources[1].append((f'q{i}',f'blue shop{i}' if i<4 else f'green shop{i}',f'{i} main road','US'))
                sources[2].append((f't{i}',f'blu shop{i}' if i<4 else f'grn shop{i}',f'{i} main rd','US'))
            for source,rows in sources.items():
                with (root/'train'/f'train_source{source}.tsv').open('w',newline='') as handle:
                    writer=csv.writer(handle,delimiter='\t')
                    writer.writerow(['entity_id','business_name','business_address','country'])
                    writer.writerows(rows)
            with (root/'train'/'train_ground_truth.tsv').open('w',newline='') as handle:
                writer=csv.writer(handle,delimiter='\t')
                writer.writerow(['source1_entity_id','matched_entity_ids'])
                writer.writerows((f'q{i}',f't{i}') for i in range(8))
            work=root/'work'
            work.mkdir()
            con=duckdb.connect()
            try:
                prepare(con,root,work,'train',legacy,work)
                aliases=json.loads((work/'aliases.json').read_text())
                self.assertEqual(aliases['name']['blu'],'blue')
                self.assertNotIn('grn',aliases['name'])
                self.assertEqual(con.execute('SELECT count(*) FROM targets').fetchone()[0],8)
            finally:
                con.close()


if __name__=='__main__':
    unittest.main()
