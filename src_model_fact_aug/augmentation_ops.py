#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----


import random
import spacy

import logging

def align_ws(old_token, new_token):
    # Align trailing whitespaces between tokens
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class Transformation():
    # Base class for all data transformations

    def __init__(self):
        # Spacy toolkit used for all NLP-related substeps
        self.spacy = spacy.load("en_core_web_sm")

    def transform(self, example):
        # Function applies transformation on passed example
        pass


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class SpeakerSwap(Transformation):
    # Swap speakers
    def __init__(self):
        super().__init__()
    
    def transform(self, example):
        ''' transform one sample '''

        new_summary, num_swap = self.__swap_speakers(example['dialogue'], example['summary'])

        if new_summary:
            new_example = { 'id': example["id"], 
                            'dialogue': example["dialogue"],
                            'summary': new_summary
                            }
            return new_example, num_swap
        else:
            return None, num_swap

    def __swap_speakers(self, dialogue_ori, summary):
        ''' swap speaker '''

        ori_sum = summary

        dialogue_ori_li = dialogue_ori.split('\n')
        speakers        = [dialogue.split(':')[0] for dialogue in dialogue_ori_li]
        speakers        = list(set(speakers))

        random.shuffle(speakers)

        swap_dict = {}
        while speakers:
            sp1 = speakers.pop()
            if speakers:
                sp2 = speakers.pop()
                swap_dict[sp1] = sp2
            else:
                continue

        for sp1, sp2 in swap_dict.items():
            summary = summary.replace(sp1, 'tmker')
            summary = summary.replace(sp2, sp1)
            summary = summary.replace('tmker', sp2)
            break

        if len(swap_dict) == 0:
            return None, 0

        if ori_sum == summary:
            return None, 0

        return summary, len(swap_dict)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class NERSwap(Transformation):
    # Swap NER objects - parent class
    def __init__(self):
        super().__init__()
        self.categories = ()

    def transform(self, example):

        dialogue = example['dialogue']
        summary  = example['summary']

        new_summary, num_swap = self.__swap_entities(dialogue, summary)

        if new_summary:
            new_example = { 'id': example["id"], 
                            'dialogue': example["dialogue"],
                            'summary': new_summary
                            }
            return new_example, num_swap
        else:
            return None, 0

    def __swap_entities(self, text, summary):

        ori_sum = summary

        dialogue_ori_li = text.split('\n')
        speakers        = [dialogue.split(':')[0] for dialogue in dialogue_ori_li]
        speakers        = list(set(speakers))

        summary_sp   = self.spacy(summary, disable=["tagger", 'lemmatizer'])
        summary_ents = [ent.text for ent in summary_sp.ents if ent.label_ in self.categories and ent.text not in speakers]

        text      = self.spacy(text, disable=["tagger", 'lemmatizer'])
        text_ents = [ent.text for ent in text.ents if ent.label_ in self.categories and ent.text not in speakers and ent.text not in summary_ents]

        swap_dict = {}
        while summary_ents:
            random.shuffle(summary_ents)
            ent1 = summary_ents.pop()
            if text_ents:
                ent2 = text_ents.pop()
                swap_dict[ent1] = ent2
            else:
                continue

        for ent1, ent2 in swap_dict.items():
            summary = summary.replace(ent1, 'tmker')
            summary = summary.replace(ent2, ent1)
            summary = summary.replace('tmker', ent2)
            break

        if summary == ori_sum:
            return None, 0

        return summary, len(swap_dict)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class EntitySwap(NERSwap):
    # NER swapping class specialized for entities (people, companies, locations, etc.)
    def __init__(self):
        super().__init__()
        self.categories = ("PERSON", "ORG", "NORP", "FAC", "GPE", "LOC", "PRODUCT",
                           "WORK_OF_ART", "EVENT")


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class DateSwap(NERSwap):
    # NER swapping class specialized for dates and time
    def __init__(self):
        super().__init__()

        self.categories = ("DATE", "TIME")


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class NumberSwap(NERSwap):
    # NER swapping class specialized for numbers (excluding dates)
    def __init__(self):
        super().__init__()

        self.categories = ("PERCENT", "MONEY", "QUANTITY", "CARDINAL")


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class PronounSwap(Transformation):
    # Swap randomly chosen pronoun
    def __init__(self, prob_swap=0.5):
        super().__init__()

        self.class2pronoun_map = {
            "SUBJECT": ["you", "he", "she", "we", "they"],
            "OBJECT": ["me", "you", "him", "her", "us", "them"],
            "POSSESSIVE": ["my", "your", "his", "her", "its", "our", "their"],
            "REFLEXIVE": ["myself", "yourself", "himself", "itself", "outselves", "yourselves", "themselves"]
        }

        self.pronoun2class_map = {pronoun: key for (key, values) in self.class2pronoun_map.items() for pronoun in values}
        self.pronouns = {pronoun for (key, values) in self.class2pronoun_map.items() for pronoun in values}

    def transform(self, example):

        # dialogue = example["dialogue"].replace("\n", " ")
        # dialogue = self.spacy(dialogue, disable=["tagger", 'lemmatizer'])
        # summary  = self.spacy(example["summary"], disable=["tagger", 'lemmatizer'])

        self.__init__()

        new_summary, num_swap = self.__swap_pronouns(example['summary'])

        if new_summary:
            new_example = { 'id': example["id"], 
                            'dialogue': example["dialogue"],
                            'summary': new_summary
                            }
            return new_example, num_swap
        else:
            return None, num_swap

    def __swap_pronouns(self, summary):

        summary_new      = summary
        summary_sp       = self.spacy(summary, disable=["tagger", 'lemmatizer'])
        summary_pronouns = [token.text for token in summary_sp if token.text.lower() in self.pronouns]
        summary_pronouns = list(set(summary_pronouns))

        if len(summary_pronouns) == 0:
            return None, 0

        for pronouns in summary_pronouns:
            chosen_class     = self.pronoun2class_map[pronouns.lower()]
            candidate_tokens = [token for token in self.class2pronoun_map[chosen_class] if token != pronouns.lower()]

            if not candidate_tokens:
                logging.info(self.class2pronoun_map[chosen_class])
                logging.info(candidate_tokens)
                break

            random.shuffle(candidate_tokens)
            rp_pronouns = candidate_tokens.pop()

            summary_new = summary_new.replace(' ' + pronouns + ' ', 'tmker')
            summary_new = summary_new.replace(' '+rp_pronouns+' ', ' '+pronouns+' ')
            summary_new = summary_new.replace('tmker', ' '+rp_pronouns+' ')

            self.class2pronoun_map[chosen_class].remove(rp_pronouns)

            break

        if summary_new == summary:
            return None, 0

        return summary_new, len(summary_pronouns)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
class NegateSentences(Transformation):
    # Apply or remove negation from negatable tokens
    def __init__(self):
        super().__init__()
        self.__negatable_tokens = ("are", "is", "was", "were", "have", "has", "had",
                                   "do", "does", "did", "can", "ca", "could", "may",
                                   "might", "must", "shall", "should", "will", "would")

    def transform(self, example):

        dialogue = example["dialogue"].replace("\n", " ")
        dialogue = self.spacy(dialogue, disable=["tagger", 'lemmatizer'])
        summary  = self.spacy(example["summary"], disable=["tagger", 'lemmatizer'])

        new_summary, num_edit = self.__negate_sentences(summary)

        if new_summary:
            new_example = { 'id': example["id"], 
                            'dialogue': example["dialogue"],
                            'summary': new_summary.text
                            }
            return new_example, num_edit
        else:
            return None, num_edit

    def __negate_sentences(self, summary):

        used_negation = []
        new_summary  = self.spacy(summary, disable=["tagger", 'lemmatizer'])

        candidate_tokens = [token for token in new_summary if token.text in self.__negatable_tokens and token.text]
        num_edit = len(candidate_tokens)


        while True:
            # find negatable token, return None if no candiates found
            candidate_tokens = [token for token in new_summary if token.text in self.__negatable_tokens and token.text not in used_negation]

            if not candidate_tokens:
                break

            # choose random token to negate
            negated_token = random.choice(candidate_tokens)
            negated_ix    = negated_token.i
            doc_len       = len(new_summary)

            if negated_ix > 0:
                if new_summary[negated_ix - 1].text in self.__negatable_tokens:
                    negated_token = new_summary[negated_ix - 1]
                    negated_ix    = negated_ix - 1
            
            # check whether token is negative
            is_negative = False
            if (doc_len - 1) > negated_ix:
                if new_summary[negated_ix + 1].text in ["not", "n't"]:
                    is_negative = True
                elif new_summary[negated_ix + 1].text == "no":
                    used_negation.append(negated_token.text)
                    continue

            # negate token
            summary_tokens = [token.text_with_ws for token in new_summary]
            if is_negative:
                if new_summary[negated_ix + 1].text.lower() == "n't":
                    if new_summary[negated_ix].text.lower() == "ca":
                        used_negation.append("can")
                        used_negation.append("ca")
                        summary_tokens[negated_ix] = "can" if summary_tokens[negated_ix].islower() else "Can"
                    summary_tokens[negated_ix] = summary_tokens[negated_ix] + " "
                summary_tokens.pop(negated_ix + 1)
        
            else:
                if new_summary[negated_ix].text.lower() in ["am", "may", "might", "must", "shall", "will"]:
                    negation = "not "
                else:
                    negation = random.choice(["not ", "n't "])

                if negation == "n't ":
                    if new_summary[negated_ix].text.lower() == "can":
                        used_negation.append("can")
                        used_negation.append("ca")
                        summary_tokens[negated_ix] = "ca" if summary_tokens[negated_ix].islower() else "Ca"
                    else:
                        summary_tokens[negated_ix] = summary_tokens[negated_ix][:-1]
                summary_tokens.insert(negated_ix + 1, negation)

            used_negation.append(negated_token.text)

            # create new claim object
            new_summary = self.spacy("".join(summary_tokens), disable=["tagger", 'lemmatizer'])

            break

        if new_summary.text == summary.text:
            return None, 0
        else:
            return new_summary, num_edit
