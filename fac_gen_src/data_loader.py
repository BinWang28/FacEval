#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


import json
import random
import logging

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq


def raw_data_loader(args):
    ''' load raw datasets '''

    logging.info("Loading data from SAMSum dataset")
    pos_data_dict, neg_data_dict = load_from_samsum(args, args.load_file)

    raw_datasets = datasets.DatasetDict({"pos_data_dict": pos_data_dict, "neg_data_dict": neg_data_dict})
    logging.info("# of Pos: {}. # of Neg: {}.".format(len(raw_datasets['pos_data_dict']), len(raw_datasets['neg_data_dict'])))

    return raw_datasets


def load_from_samsum(args, file_path):
    ''' load samsum jsonl data '''

    pos_id_list       = []
    pos_dialogue_list = []
    pos_summary_list  = []
    pos_marker_list   = []

    neg_id_list       = []
    neg_dialogue_list = []
    neg_summary_list  = []
    neg_marker_list   = []

    with open(file_path, 'r') as f:
        samples = json.load(f)

    for sample in samples:

        if sample['label'] == 'pos':
            pos_id_list.append(sample['id'])
            pos_dialogue_list.append(sample['dialogue'])
            pos_summary_list.append(sample['summary'])
            pos_marker_list.append(sample['label'])

        elif args.neg_marker in sample['label']:
            neg_id_list.append(sample['id'])
            neg_dialogue_list.append(sample['dialogue'])
            neg_summary_list.append(sample['summary'])
            neg_marker_list.append(sample['label'])

    pos_data_dict = {
                    'id'      : pos_id_list,
                    'dialogue': pos_dialogue_list,
                    'summary' : pos_summary_list,
                    'marker'  : pos_marker_list
                    }

    neg_data_dict = {
                    'id'      : neg_id_list,
                    'dialogue': neg_dialogue_list,
                    'summary' : neg_summary_list,
                    'marker'  : neg_marker_list
                    }

    pos_data_dict = Dataset.from_dict(pos_data_dict)
    neg_data_dict = Dataset.from_dict(neg_data_dict)

    return pos_data_dict, neg_data_dict


def data_processor(logger, args, accelerator, raw_datasets, tokenizer, model):
    ''' prepare dataset format for train / val / test '''

    def preprocess_function(examples):

        # summary - target
        targets = examples[summary_column]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        if 'neg_summary' in examples:
            with tokenizer.as_target_tokenizer():
                neg_summary = tokenizer(examples['neg_summary'], max_length=max_target_length, padding=padding, truncation=True)

        # dialogue - input
        inputs = examples[text_column]
        new_inputs = []
        for i, inp in enumerate(inputs):
            new_inputs.append(prefix + inp)

        inputs       = new_inputs
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        if 'neg_summary' in examples:
            model_inputs["neg_summary"] = neg_summary["input_ids"]

        return model_inputs


    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["pos_data_dict"].column_names


    # Get the column names for input/target.
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    summary_column = args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    pos_dataset = processed_datasets["pos_data_dict"]
    neg_dataset  = processed_datasets["neg_data_dict"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(pos_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {pos_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    pos_dataloader  = DataLoader(pos_dataset, collate_fn=data_collator, batch_size=1)
    neg_dataloader  = DataLoader(neg_dataset, collate_fn=data_collator, batch_size=1)

    return (pos_dataloader, neg_dataloader), (pos_dataset, neg_dataset)

