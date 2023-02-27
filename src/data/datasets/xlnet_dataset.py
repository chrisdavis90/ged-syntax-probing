import numpy as np

import torch
from data.datasets import BaseDataSet


class XLNetDataSet(BaseDataSet):

    def get_max_len(self):
        maxlen = super().get_max_len()
        #  +2 for special tokens
        return maxlen + 2

    def __getitem__(self, idx):

        '''
            From: https://huggingface.co/docs/transformers/model_doc/xlnet#transformers.XLNetTokenizer

            Build model inputs from a sequence or a pair of sequence
                 for sequence classification tasks by concatenating and
                 adding special tokens. An XLNet sequence has the following format:

            single sequence: X <sep> <cls>
            pair of sequences: A <sep> B <sep> <cls>
        '''

        input_example = self.data_list[idx]
        text = input_example.text
        labels = input_example.label
        
        sub_tokens = []
        sub_token_indices = []
        label_list = []
        label_mask = []

        input_ids = []
        label_ids = []

        # iterate over individual tokens and their labels
        for i in range(len(text)):
            token = text[i]
            label = labels[i]

            tokenized_token, num_sub_tokens = self.tokenize(token, i)
            tokenized_word_check = self.tokenizer.encode(text=self.tokenizer.tokenize(token),add_special_tokens=False)

            for t in tokenized_token:
                sub_tokens.append(t)
                sub_token_indices.append(i)
                # input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            if len(tokenized_token) == 0:
                print("WARNING: There is a word that the tokenizer returned nothin for:", token)
                print(f'\t substituted the word with {self.tokenizer.unk_token}')
                sub_tokens.append(self.tokenizer.unk_token)
                sub_token_indices.append(-1)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token))
            
            label_list.append(label)
            label_ids.append(self.label_map[label])
            label_mask.append(1)
            
            for i in range(1, num_sub_tokens):
                label_list.append(label)
                label_ids.append(self.label_map[label])
                label_mask.append(0)
        
        # this encodes the tokens to input_ids
        encoded_inputs = self.tokenizer.encode(
            text=sub_tokens, add_special_tokens=False)

        # This pads the inputs to max_len using a the ID for the padding token <pad> as well as 
        #  truncating any inputs longer than max_len.
        # It also returns an attention mask, token_type_ids, and a special_tokens_mask (although we don't use this now)
        model_data = self.tokenizer.prepare_for_model(
            encoded_inputs,
            return_special_tokens_mask=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
            # pad_to_max_length=True
        )

        # manually truncate label_ids to max_len - 2 (to make room for 2 padding labels to correspond to the special tokens)
        # Otherwise, pad label_ids to max_len
        # todo: WIP. Clean this up.
        if len(label_ids) > (self.max_len - 2):
            sub_token_indices = sub_token_indices[:(self.max_len - 2)]
            sub_token_indices.extend([-1, -1])
            
            sub_tokens = sub_tokens[:(self.max_len -2)]
            sub_tokens.extend([self.tokenizer.pad_token, self.tokenizer.pad_token])

            label_ids = label_ids[:(self.max_len - 2)]
            label_ids.extend([self.label_map['C'], self.label_map['C']]) # for <sep> and <cls> tokens

            label_mask = label_mask[:(self.max_len - 2)]
            label_mask.extend([0, 0])

            label_list = label_list[:(self.max_len - 2)]
            label_list.extend(['C', 'C'])
        else:
            # xlnet pads at the start of the sentence
            # pad + original + special tokens
            label_ids = ([self.label_map['C']] * ((self.max_len - 2) - len(label_ids))) + label_ids + [self.label_map['C'], self.label_map['C']]
            sub_token_indices = ([-1] * ((self.max_len - 2) - len(sub_token_indices))) + sub_token_indices + [-1, -1]
            sub_tokens = ([self.tokenizer.pad_token] * ((self.max_len - 2) - len(sub_tokens))) + sub_tokens + [self.tokenizer.pad_token, self.tokenizer.pad_token]
            label_mask = ([0] * ((self.max_len - 2) - len(label_mask))) + label_mask + [0, 0]
            label_list = (['C'] * ((self.max_len - 2) - len(label_list))) + label_list + ['C', 'C']

            # xlnet pads at the start of the sentence but we set it to pad
            #  on the right
            # original + special tokens + padding
            # while len(label_ids) < self.max_len:
            #     sub_token_indices.append(-1)
            #     sub_tokens.append(self.tokenizer.pad_token)
            #     label_ids.append(self.label_map['C'])
            #     label_mask.append(0)
            #     label_list.append('C')

            # label_ids = ([self.label_map['C']] * ((self.max_len - 2) - len(label_ids))) + label_ids + [self.label_map['C'], self.label_map['C']]
            # sub_token_indices = ([-1] * ((self.max_len - 2) - len(sub_token_indices))) + sub_token_indices + [-1, -1]
            # sub_tokens = ([self.tokenizer.pad_token] * ((self.max_len - 2) - len(sub_tokens))) + sub_tokens + [self.tokenizer.pad_token, self.tokenizer.pad_token]
            # label_mask = ([0] * ((self.max_len - 2) - len(label_mask))) + label_mask + [0, 0]
            # label_list = (['C'] * ((self.max_len - 2) - len(label_list))) + label_list + ['C', 'C']

        assert len(model_data['input_ids']) == len(label_ids) == len(label_list) == len(label_mask) == len(sub_tokens) == len(sub_token_indices) <= self.max_len, idx

        attention_mask = model_data['attention_mask']
        token_type_ids = model_data['token_type_ids']
        input_ids = model_data['input_ids']
        
        # padding elements
        padlen = len(attention_mask) - sum(attention_mask)
        '''
        perm_mask (torch.FloatTensor of shape (batch_size, sequence_length, sequence_length), optional) — Mask to indicate the attention pattern for each input token with values selected in [0, 1]:
        if perm_mask[k, i, j] = 0, i attend to j in batch k;
        if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
        If not set, each token attends to all the others (full bidirectional attention). Only used during pretraining (to define factorization order) or for sequential decoding (generation).
        '''
        if self.unidirectional:
            perm_mask = np.ones((len(encoded_inputs)+2, len(encoded_inputs)+2))
            perm_mask_triu = np.triu(perm_mask, 1)
            
            # add padding (start of sentence) to perm_mask
            perm_mask_col_padding = np.ones((perm_mask.shape[0], padlen))
            perm_mask_triu = np.concatenate((perm_mask_col_padding, perm_mask_triu), axis=1)
            perm_mask_row_padding = np.ones((padlen, perm_mask_triu.shape[1]))
            perm_mask_triu = np.concatenate((perm_mask_row_padding, perm_mask_triu), axis=0)

            assert perm_mask_triu.shape[0] == perm_mask_triu.shape[1] == len(attention_mask)
        else:
            perm_mask_triu = np.zeros((len(attention_mask), len(attention_mask)))

        return torch.LongTensor(sub_token_indices), sub_tokens, label_list, torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids), torch.BoolTensor(label_mask), torch.LongTensor(perm_mask_triu)