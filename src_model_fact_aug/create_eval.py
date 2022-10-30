#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import json
import logging

from tqdm.contrib.concurrent import process_map
import augmentation_ops as ops

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# load samsum test
file_path   = './data/samsum.test.jsonl'
raw_samples = []
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        sample = json.loads(line)
        raw_samples.append(sample)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# load samsum test back traslation
file_path   = './data/samsum.test.bt.jsonl'
bt_samples = []
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        sample = json.loads(line)
        bt_samples.append(sample)

all_pos_sample = raw_samples + bt_samples

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# perform Speaker Swap
def _ss_corrupt(sample):
    ''' Speaker swap '''
    sample = sample.copy()
    speakerswap_op = ops.SpeakerSwap()
    sample_ss, _ = speakerswap_op.transform(sample)
    return sample_ss, sample

print("perform Speaker Swap * 5 times")

ss_samples = process_map(_ss_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ss_samples2 = process_map(_ss_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ss_samples3 = process_map(_ss_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ss_samples4 = process_map(_ss_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ss_samples5 = process_map(_ss_corrupt, all_pos_sample, max_workers=10, chunksize=1)


ss_samples = [item[0] for item in ss_samples if item[0] is not None]
ss_samples2 = [item[0] for item in ss_samples2 if item[0] is not None]
ss_samples3 = [item[0] for item in ss_samples3 if item[0] is not None]
ss_samples4 = [item[0] for item in ss_samples4 if item[0] is not None]
ss_samples5 = [item[0] for item in ss_samples5 if item[0] is not None]

for item in ss_samples2 + ss_samples3 + ss_samples4 + ss_samples5:
    if item not in ss_samples:
        ss_samples.append(item)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# perform Entity Swap
def _es_corrupt(sample):
    ''' Entity swap '''
    sample = sample.copy()
    entswap_op = ops.EntitySwap()
    sample_es, _ = entswap_op.transform(sample)
    return sample_es, sample

print("perform Entity Swap * 5 times")

es_samples = process_map(_es_corrupt, all_pos_sample, max_workers=10, chunksize=1)
es_samples2 = process_map(_es_corrupt, all_pos_sample, max_workers=10, chunksize=1)
es_samples3 = process_map(_es_corrupt, all_pos_sample, max_workers=10, chunksize=1)
es_samples4 = process_map(_es_corrupt, all_pos_sample, max_workers=10, chunksize=1)
es_samples5 = process_map(_es_corrupt, all_pos_sample, max_workers=10, chunksize=1)

es_samples = [item[0] for item in es_samples if item[0] is not None]
es_samples2 = [item[0] for item in es_samples2 if item[0] is not None]
es_samples3 = [item[0] for item in es_samples3 if item[0] is not None]
es_samples4 = [item[0] for item in es_samples4 if item[0] is not None]
es_samples5 = [item[0] for item in es_samples5 if item[0] is not None]

for item in es_samples2 + es_samples3 + es_samples4 + es_samples5:
    if item not in es_samples:
        es_samples.append(item)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# perform Date Swap
def _ds_corrupt(sample):
    ''' Date swap '''
    sample = sample.copy()
    dateswap_op = ops.DateSwap()
    sample_ds, _ = dateswap_op.transform(sample)
    return sample_ds, sample

print("perform Date Swap * 5 times")

ds_samples = process_map(_ds_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ds_samples2 = process_map(_ds_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ds_samples3 = process_map(_ds_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ds_samples4 = process_map(_ds_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ds_samples5 = process_map(_ds_corrupt, all_pos_sample, max_workers=10, chunksize=1)

ds_samples = [item[0] for item in ds_samples if item[0] is not None]
ds_samples2 = [item[0] for item in ds_samples2 if item[0] is not None]
ds_samples3 = [item[0] for item in ds_samples3 if item[0] is not None]
ds_samples4 = [item[0] for item in ds_samples4 if item[0] is not None]
ds_samples5 = [item[0] for item in ds_samples5 if item[0] is not None]

for item in ds_samples2 + ds_samples3 + ds_samples4 + ds_samples5:
    if item not in ds_samples:
        ds_samples.append(item)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# perform Number Swap
def _ns_corrupt(sample):
    ''' Number swap '''
    sample = sample.copy()
    numbswap_op = ops.NumberSwap()
    sample_ns, _ = numbswap_op.transform(sample)
    return sample_ns, sample

print("perform Number Swap * 5 times")

ns_samples = process_map(_ns_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ns_samples2 = process_map(_ns_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ns_samples3 = process_map(_ns_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ns_samples4 = process_map(_ns_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ns_samples5 = process_map(_ns_corrupt, all_pos_sample, max_workers=10, chunksize=1)

ns_samples = [item[0] for item in ns_samples if item[0] is not None]
ns_samples2 = [item[0] for item in ns_samples2 if item[0] is not None]
ns_samples3 = [item[0] for item in ns_samples3 if item[0] is not None]
ns_samples4 = [item[0] for item in ns_samples4 if item[0] is not None]
ns_samples5 = [item[0] for item in ns_samples5 if item[0] is not None]

for item in ns_samples2 + ns_samples3 + ns_samples4 + ns_samples5:
    if item not in ns_samples:
        ns_samples.append(item)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# perform Pronoun Swap
def _ps_corrupt(sample):
    ''' Pronoun swap '''
    sample = sample.copy()
    pronswap_op = ops.PronounSwap()
    sample_ps, _ = pronswap_op.transform(sample)
    return sample_ps, sample

print("perform Pronoun Swap * 5 times")

ps_samples = process_map(_ps_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ps_samples2 = process_map(_ps_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ps_samples3 = process_map(_ps_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ps_samples4 = process_map(_ps_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ps_samples5 = process_map(_ps_corrupt, all_pos_sample, max_workers=10, chunksize=1)

ps_samples = [item[0] for item in ps_samples if item[0] is not None]
ps_samples2 = [item[0] for item in ps_samples2 if item[0] is not None]
ps_samples3 = [item[0] for item in ps_samples3 if item[0] is not None]
ps_samples4 = [item[0] for item in ps_samples4 if item[0] is not None]
ps_samples5 = [item[0] for item in ps_samples5 if item[0] is not None]

for item in ps_samples2 + ps_samples3 + ps_samples4 + ps_samples5:
    if item not in ps_samples:
        ps_samples.append(item)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# perform Negation
def _ng_corrupt(sample):
    ''' Negation '''
    sample = sample.copy()
    negate_op = ops.NegateSentences()
    sample_ng, _ = negate_op.transform(sample)
    return sample_ng, sample

print("perform Negation * 5 times")

ng_samples = process_map(_ng_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ng_samples2 = process_map(_ng_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ng_samples3 = process_map(_ng_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ng_samples4 = process_map(_ng_corrupt, all_pos_sample, max_workers=10, chunksize=1)
ng_samples5 = process_map(_ng_corrupt, all_pos_sample, max_workers=10, chunksize=1)

ng_samples = [item[0] for item in ng_samples if item[0] is not None]
ng_samples2 = [item[0] for item in ng_samples2 if item[0] is not None]
ng_samples3 = [item[0] for item in ng_samples3 if item[0] is not None]
ng_samples4 = [item[0] for item in ng_samples4 if item[0] is not None]
ng_samples5 = [item[0] for item in ng_samples5 if item[0] is not None]

for item in ng_samples2 + ng_samples3 + ng_samples4 + ng_samples5:
    if item not in ng_samples:
        ng_samples.append(item)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# all_pos_sample
all_samples = []
for sample in all_pos_sample:
    sample['label'] = 'pos'
    all_samples.append(sample)
    
for sample in ss_samples:
    sample['label'] = 'ss_neg'
    all_samples.append(sample)


for sample in es_samples:
    sample['label'] = 'es_neg'
    all_samples.append(sample)


for sample in ds_samples:
    sample['label'] = 'ds_neg'
    all_samples.append(sample)


for sample in ns_samples:
    sample['label'] = 'ns_neg'
    all_samples.append(sample)


for sample in ps_samples:
    sample['label'] = 'ps_neg'
    all_samples.append(sample)


for sample in ng_samples:
    sample['label'] = 'ng_neg'
    all_samples.append(sample)


with open('./new_generated_samples.jsonl', 'w') as f:
    json.dump(all_samples, f, indent=4)