#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data_a12/binwang/research/FacEval/testing_source/data/clean.py
# Project: /data_a12/binwang/research/FacEval/testing_source/data
# Created Date: 1970-01-01 07:30:00
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###



import json

with open('faceval_samples_old.jsonl', 'r') as f:
    lines = f.readlines()

all_samples = {}

for line in lines:
    sample = json.loads(line)

    id_name  = sample['id']
    dialogue = sample['dialogue']
    summary  = sample['summary']
    label    = sample['label']

    if id_name not in all_samples:
        all_samples[id_name] = []
        all_samples[id_name].append([dialogue, summary, label])

    elif id_name in all_samples:
        all_samples[id_name].append([dialogue, summary, label])

# Save it as json

all_samples_json = []
for sample_id, samples in all_samples.items():

    for sample in samples:

        temp_sample = {
            'id': sample_id,
            'dialogue': sample[0],
            'summary': sample[1],
            'label': sample[2]
        }

        all_samples_json.append(temp_sample)



with open('faceval_sample.json', 'w') as outfile:
    json.dump(all_samples_json, outfile, indent=4, ensure_ascii=False)


