#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


import os
import logging

import nltk
import numpy as np
import torch
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed

from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

from args import parse_args
from data_loader import raw_data_loader, data_processor
from model_loader import model_loader


# =  =  =  =  =  =  =  =  =  = Logging Setup =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# =  =  =  =  =  =  =  =  =  = Pre-check Package Info =  =  =  =  =  =  =  =  =  =  =  = 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# = = = = = = = = = = = = = Main Process = = = = = = = = = = = = = = = = = =
def main():
    args = parse_args()
    
    # Display Parameters
    logging.info("*** Parameters ***")
    for item, value in vars(args).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")

    # Initialize the accelerator. The accelerator will handle device placement for us.
    accelerator = Accelerator()
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        #datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        #datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # load raw dataset
    raw_datasets = raw_data_loader(args)

    # load model (config, tokenizer, s2s model)
    config, tokenizer, model = model_loader(accelerator, logger, args)
    
    # data processor (for DataLoader)
    dataloader    , processed_dataset = data_processor(logger, args, accelerator, raw_datasets, tokenizer, model)
    pos_dataloader, neg_dataloader    = dataloader

    model, pos_dataloader, neg_dataloader = accelerator.prepare(model, pos_dataloader, neg_dataloader)


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = Extrac Generation Probabilities =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
    model.eval()

    # compute positive probability
    index            = 0
    pos_sample_probs = {}
    id_list          = raw_datasets['pos_data_dict']['id']
    for _, batch in enumerate(tqdm(pos_dataloader)):
        outputs       = model(**batch)
        output_logits = outputs.logits
        output_probs  = torch.nn.functional.softmax(output_logits, dim=-1)
        gt_logits     = batch['labels']

        lprobs, target = output_probs, gt_logits

        ignore_index = -100
        lprobs       = lprobs[~target.eq(ignore_index)]
        target       = target[~target.eq(ignore_index)]

        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)

        predict_prob = lprobs.gather(dim=-1, index=target)
        predict_prob = predict_prob.cpu().detach().numpy()
        predict_prob = np.log(predict_prob)
        predict_prob = predict_prob.mean()
        predict_prob = float(predict_prob)

        if id_list[index] not in pos_sample_probs:
            pos_sample_probs[id_list[index]] = [predict_prob]
        else:
            pos_sample_probs[id_list[index]].append(predict_prob)
        index += 1

    # Compute negative probability
    index            = 0
    neg_sample_probs = {}
    id_list          = raw_datasets['neg_data_dict']['id']
    for _, batch in enumerate(tqdm(neg_dataloader)):
        outputs       = model(**batch)
        output_logits = outputs.logits
        output_probs  = torch.nn.functional.softmax(output_logits, dim=-1)
        gt_logits     = batch['labels']

        lprobs, target = output_probs, gt_logits

        ignore_index = -100
        lprobs       = lprobs[~target.eq(ignore_index)]
        target       = target[~target.eq(ignore_index)]

        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)

        predict_prob = lprobs.gather(dim=-1, index=target)
        predict_prob = predict_prob.cpu().detach().numpy()
        predict_prob = np.log(predict_prob)
        predict_prob = predict_prob.mean()
        predict_prob = float(predict_prob)

        if id_list[index] not in neg_sample_probs:
            neg_sample_probs[id_list[index]] = [predict_prob]
        else:
            neg_sample_probs[id_list[index]].append(predict_prob)
        index += 1

    # Compute average scores
    scores = []
    for key, values in pos_sample_probs.items():
        positive_values = values

        if key in neg_sample_probs:
            negative_values = neg_sample_probs[key]
        else:
            continue

        all_comparison = []
        for pos_value in positive_values:
            for neg_value in negative_values:
                if pos_value > neg_value: 
                    all_comparison.append(1) 
                else:
                    all_comparison.append(0)

        scores.append(np.mean(all_comparison))
    
    final_score = np.mean(scores)

    with open(args.output_dir + '/' + args.neg_marker + '_score.txt', 'w') as f:
        f.write(str(final_score))
        

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Main Process
if __name__ == "__main__":
    main()
