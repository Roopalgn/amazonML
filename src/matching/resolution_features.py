"""Multilingual, corpus-aware evidence. Learned aliases use fitting labels only."""

from collections import Counter, defaultdict
from functools import lru_cache
import math
import re

from anyascii import anyascii
import numpy as np
from rapidfuzz import fuzz, process

from src.matching.pair_features import FEATURE_NAMES, pair_features


VERSION = 'resolution-evidence-v2'
LEGAL = frozenset('inc incorporated corporation corp company co limited ltd private pvt llc llp plc sarl sas sa'.split())
EXTRA = frozenset('id ref registration registered enterprises enterprise services service center group official online com www'.split())
ADDRESS_ABBREVIATIONS = dict(zip(
    'street road avenue boulevard drive lane court place apartment suite floor building highway north south east west'.split(),
    'st rd ave blvd dr ln ct pl apt ste fl bldg hwy n s e w'.split()))


@lru_cache(maxsize=100000)
def latin(value):
    # Transliterate the original string; stripping marks first destroys Indic vowels.
    value = anyascii(value or '').lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', value)).strip()


def clean_name(value):
    tokens = latin(value).split()
    tokens = [t for t in tokens if t not in LEGAL and t not in EXTRA and not t.isdigit()]
    return ' '.join(tokens) or latin(value)


def clean_address(value):
    value = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', latin(value))
    return ' '.join(str(int(t)) if t.isdigit() else ADDRESS_ABBREVIATIONS.get(t, t)
                    for t in value.split())


def learn_aliases(pairs, minimum=4):
    """Align tokens by repeated positive-pair evidence, excluding unchanged words.

    Repeated variants of one business count once, preventing a single entity from
    inventing a translation. Both probability and association must support a map.
    """
    counts = defaultdict(Counter)
    source_counts = Counter()
    target_counts = Counter()
    seen_source, seen_target, seen_links = set(), set(), set()
    for qid, source, target in pairs:
        left, right = set(source.split()), set(target.split())
        shared = left & right
        left, right = left-shared, right-shared
        for token in left:
            if (qid,token) not in seen_source:
                source_counts[token]+=1
                seen_source.add((qid,token))
        for token in right:
            if (qid,token) not in seen_target:
                target_counts[token]+=1
                seen_target.add((qid,token))
        for token in right:
            if token.isdigit() or len(token)<2:
                continue
            for t in left:
                if not t.isdigit() and len(t)>=2 and (qid,token,t) not in seen_links:
                    counts[token][t]+=1
                    seen_links.add((qid,token,t))
    aliases = {}
    for token, options in counts.items():
        ranked = sorted(options.items(), key=lambda p: p[1] / math.sqrt(source_counts[p[0]]), reverse=True)
        if not ranked:
            continue
        best, support = ranked[0]
        confidence = support / target_counts[token]
        association = support / math.sqrt(source_counts[best])
        runner_up = ranked[1][1] / math.sqrt(source_counts[ranked[1][0]]) if len(ranked)>1 else 0
        if support>=minimum and confidence>=.60 and association>1.5*runner_up:
            aliases[token] = best
    return aliases


def mapped(value, aliases):
    return ' '.join(aliases.get(t,t) for t in value.split())


def skeleton(value):
    return re.sub(r'(.)\1+', r'\1', re.sub('[aeiou]', '', value.replace(' ','')))


EXTRA_NAMES = [
    'core_ratio','core_sorted','core_set','core_partial',
    'canonical_address_ratio','canonical_address_sorted','canonical_address_set','canonical_address_partial',
    'name_skeleton','name_initials','compact_core','cross_name_address',
    'name_idf_recall','name_idf_precision','name_idf_jaccard','name_shared_rarity',
    'address_idf_recall','address_idf_precision','address_idf_jaccard','address_shared_rarity',
    'name_fuzzy_recall','name_fuzzy_precision','name_fuzzy_min',
    'address_fuzzy_recall','address_fuzzy_precision','address_fuzzy_min',
    'number_recall','number_precision','first_number_in_other','other_first_number_in_query',
    'name_frequency','address_frequency','core_length_ratio','canonical_address_length_ratio',
]
ALL_FEATURE_NAMES = FEATURE_NAMES + EXTRA_NAMES


def weighted_overlap(left, right, weights):
    left, right = set(left), set(right)
    weight = lambda token: weights.get(token, 10.)
    shared = sum(weight(t) for t in left & right)
    first, second = sum(weight(t) for t in left), sum(weight(t) for t in right)
    return [shared/max(first,1),shared/max(second,1),shared/max(first+second-shared,1),
            max((weight(t) for t in left & right),default=0)]


def fuzzy_overlap(left, right, weights):
    if not left or not right:
        return [0.,0.,0.]
    matrix = process.cdist(left,right,scorer=fuzz.ratio,dtype=np.float32,workers=1)/100
    lw = np.array([weights.get(t,10.) for t in left])
    rw = np.array([weights.get(t,10.) for t in right])
    return [np.average(matrix.max(axis=1),weights=lw),
            np.average(matrix.max(axis=0),weights=rw),float(matrix.max(axis=1).min())]


def fuzzy_batch(lefts, rights, weights, chunk_size=500):
    """Batch the small token matrices to avoid millions of Python/NumPy calls."""
    result=np.zeros((len(lefts),3),dtype=np.float32)
    for start in range(0,len(lefts),chunk_size):
        size=min(chunk_size,len(lefts)-start)
        ls,rs,li,ri=[],[],[],[]
        lw,rw,lrows,rrows=[],[],[],[]
        for i,(left,right) in enumerate(zip(lefts[start:start+size],rights[start:start+size])):
            lt,rt=list(dict.fromkeys(left.split())),list(dict.fromkeys(right.split()))
            if not lt or not rt:
                continue
            lo,ro=len(lw),len(rw)
            rindices=list(range(ro,ro+len(rt)))
            for j,token in enumerate(lt):
                ls.extend([token]*len(rt))
                rs.extend(rt)
                li.extend([lo+j]*len(rt))
                ri.extend(rindices)
            lw.extend(weights.get(t,10.) for t in lt)
            rw.extend(weights.get(t,10.) for t in rt)
            lrows.extend([i]*len(lt))
            rrows.extend([i]*len(rt))
        if not ls:
            continue
        values=process.cpdist(ls,rs,scorer=fuzz.ratio,dtype=np.float32,workers=2)/100
        lm,rm=np.zeros(len(lw),dtype=np.float32),np.zeros(len(rw),dtype=np.float32)
        np.maximum.at(lm,np.asarray(li,dtype=np.int32),values)
        np.maximum.at(rm,np.asarray(ri,dtype=np.int32),values)
        lrows,rrows=np.asarray(lrows,dtype=np.int32),np.asarray(rrows,dtype=np.int32)
        lweight,rweight=np.asarray(lw),np.asarray(rw)
        result[start:start+size,0]=np.bincount(lrows,weights=lm*lweight,minlength=size)/np.maximum(1e-10,np.bincount(lrows,weights=lweight,minlength=size))
        result[start:start+size,1]=np.bincount(rrows,weights=rm*rweight,minlength=size)/np.maximum(1e-10,np.bincount(rrows,weights=rweight,minlength=size))
        minima=np.ones(size,dtype=np.float32)
        np.minimum.at(minima,lrows,lm)
        minima[np.bincount(lrows,minlength=size)==0]=0
        result[start:start+size,2]=minima
    return result


def features(rows, name_weights=None, address_weights=None):
    """Rows: qid,tid,legacy qname,qaddress,tname,taddress,canonical names/addresses,
    followed by q-name frequency and q-address frequency. IDs never enter features.
    """
    if not rows:
        return np.empty((0,len(ALL_FEATURE_NAMES)),dtype=np.float32)
    name_weights, address_weights = name_weights or {}, address_weights or {}
    base = pair_features(rows)
    n,a,tn,ta = ([r[i] or '' for r in rows] for i in range(6,10))
    result = []
    for left,right in ((n,tn),(a,ta)):
        for scorer in (fuzz.ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio,fuzz.partial_ratio):
            result.append(process.cpdist(left,right,scorer=scorer,dtype=np.float32,workers=2)/100)
    for left,right in (([skeleton(v) for v in n],[skeleton(v) for v in tn]),
                       ([''.join(t[0] for t in v.split()) for v in n],
                        [''.join(t[0] for t in v.split()) for v in tn]),
                       ([v.replace(' ','') for v in n],[v.replace(' ','') for v in tn]),
                       (n,ta)):
        result.append(process.cpdist(left,right,scorer=fuzz.ratio,dtype=np.float32,workers=2)/100)
    fuzzy_names=fuzzy_batch(n,tn,name_weights)
    fuzzy_addresses=fuzzy_batch(a,ta,address_weights)
    extra = []
    for i,(row,name,address,other_name,other_address) in enumerate(zip(rows,n,a,tn,ta)):
        nt,at,tnt,tat = [list(dict.fromkeys(s.split())) for s in (name,address,other_name,other_address)]
        nums,tnums = re.findall(r'\d+',address),re.findall(r'\d+',other_address)
        overlap = len(set(nums)&set(tnums))
        extra.append([
            *weighted_overlap(nt,tnt,name_weights),*weighted_overlap(at,tat,address_weights),
            *fuzzy_names[i],*fuzzy_addresses[i],
            overlap/max(1,len(set(nums))),overlap/max(1,len(set(tnums))),
            bool(nums) and nums[0] in tnums,bool(tnums) and tnums[0] in nums,
            math.log1p(row[10]),math.log1p(row[11]),
            min(len(name),len(other_name))/max(1,len(name),len(other_name)),
            min(len(address),len(other_address))/max(1,len(address),len(other_address)),
        ])
    return np.column_stack([base,*result,np.asarray(extra,dtype=np.float32)]).astype(np.float32)
