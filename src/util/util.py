# from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import itertools
import json
import os
import pickle
import random
import sys
from enum import Enum

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import (BertTokenizer, ElectraTokenizer, GPT2Tokenizer,
                          RobertaTokenizer, XLNetTokenizer)

from .enum_ged_label import GEDLabel


def allowed_enum_items(e: Enum):
    choices = [item.name for item in e]
    return choices

def config_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def from_file(data_path: str, layer: int=None):
    '''
        read files saved in to_file():
            tokens.pickle label_masks.pickle  labels.pickle  reps.npy
        or
            tokens.pickle label_masks.pickle  labels.pickle  reps_layer:N.npy
                where N = layer number
    '''

    tokens_f = os.path.join(data_path, 'tokens.pickle')
    label_masks_f = os.path.join(data_path, 'label_masks.pickle')
    labels_f = os.path.join(data_path, 'labels.pickle')
    subtoken_indices_f = os.path.join(data_path, 'sub_token_indices.pickle')

    with open(tokens_f, 'rb') as f:
        tokens = pickle.load(f)
    with open(label_masks_f, 'rb') as f:
        label_masks = pickle.load(f)
    with open(labels_f, 'rb') as f:
        labels = pickle.load(f)
    with open(subtoken_indices_f, 'rb') as f:
        subword_indices = pickle.load(f)

    if layer is None:
        reps_f = os.path.join(data_path, 'reps.npy')
    else:
        reps_f = os.path.join(data_path, f'reps_layer:{layer}.npy')

    reps = np.load(reps_f, allow_pickle=True)

    assert len(reps) == len(labels) == len(tokens) == len(label_masks) == len(subword_indices)

    for i in range(len(reps)):
        assert len(reps[i]) == len(labels[i]) == len(tokens[i]) == len(label_masks[i]) == len(subword_indices[i])

    return reps, tokens, labels, label_masks, subword_indices


def to_file(encoded_data, output_dir, only_last_layer):
    if not os.path.isdir(output_dir):
        print(f'Creating directory: {output_dir}')
        os.makedirs(output_dir)

    if not only_last_layer:
        for layer in range(len(encoded_data['vectors'][0])):
            # get a list of representations for a specific layer
            # e.g. layer 2 representations from the whole dataset
            X = np.array([x[layer] for i, x in enumerate(encoded_data['vectors'])])

            np.save(os.path.join(output_dir, f'reps_layer:{layer}.npy'), X)
    else:
        np.save(os.path.join(output_dir, 'reps.npy'), np.array(encoded_data['vectors']))

    for name, vals in encoded_data['labels'].items():
        with open(os.path.join(output_dir, f'{name}.pickle'), 'wb') as f:
            pickle.dump(vals, f)


def create_tokenizer(model_name, saved_path_or_model_name):
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(saved_path_or_model_name)
    elif 'bert' in model_name:
        tokenizer = BertTokenizer.from_pretrained(saved_path_or_model_name)
    elif 'xlnet' in model_name:
        # tokenizer = XLNetTokenizer.from_pretrained(
        #     saved_path_or_model_name, padding_side='right')
        tokenizer = XLNetTokenizer.from_pretrained(
            saved_path_or_model_name)
    elif 'electra' in model_name:
        tokenizer = ElectraTokenizer.from_pretrained(saved_path_or_model_name)
    elif 'gpt2' in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(saved_path_or_model_name)
    else:
        print(f'"{model_name}" is not a recognised model name.')
        tokenizer = None

    return tokenizer


def _is_divider(line: str) -> bool:
    return line.strip() == ''


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def read_json(config_file: str) -> dict:
    with open(config_file) as f:
        data = json.load(f)

    return data


def format_error(error: str, ged_label: GEDLabel):
    '''
        e.g. error = R:VERB:SVA

        GEDLabel.BINARY:
            error = I
        GEDLabel.OP:
            error = R
        GEDLabel.MAIN
            error = VERB:SVA
        GEDLabel.ALL:
            error = R:VERB:SVA
    '''

    if error.upper() == 'C':
        return 'C'
    
    if ged_label == GEDLabel.BINARY:
        return 'I'
    elif ged_label == GEDLabel.OP:
        return error[0]
    elif ged_label == GEDLabel.MAIN:
        return error.split('-')[0][2:]
    else:
        return error.split('-')[0]


def readfile_singlecolumn(file: str, ged_label: GEDLabel):

    data = []
    with open(file, "r") as data_file:
        # Group into alternative divider / sentence chunks.
        for is_divider, lines_grouper in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words of a single sentence.
            if not is_divider:

                lines = list(lines_grouper)

                fields = [line.strip().split() for line in lines]
                fields = [list(field) for field in zip(*fields)]

                # extract labels and tokens
                tokens = [convert_to_unicode(token) for token in fields[0]]
                labels = [x.upper() for x in fields[-1]]

                # process labels
                labels = [
                    format_error(error=label, ged_label=ged_label)
                    for label in labels
                ]

                data.append((tokens, labels))
 
    return data


def readfile_multicolumn(file: str, ged_label: GEDLabel):

    data = []
    with open(file, "r") as data_file:
        # Group into alternative divider / sentence chunks.
        for is_divider, lines_grouper in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words of a single sentence.
            if not is_divider:

                lines = list(lines_grouper)

                fields = [line.strip().split() for line in lines]
                fields = [list(field) for field in zip(*fields)]

                # extract labels and tokens
                tokens = [convert_to_unicode(token) for token in fields[0]]
                binary_labels = [x.upper() for x in fields[-1]]
                fine_grained_labels = fields[-2]

                # process labels
                if ged_label == GEDLabel.BINARY:
                    labels = binary_labels
                else:
                    labels = []
                    for bl, fl in zip(binary_labels, fine_grained_labels):
                        if bl == 'C':
                            labels.append(bl)
                        else:
                            labels.append(format_error(error=fl, ged_label=ged_label))
                    
                data.append((tokens, labels))
                
    return data

