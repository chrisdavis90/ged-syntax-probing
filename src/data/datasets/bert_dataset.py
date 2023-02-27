import torch
from data.datasets import BaseDataSet


class BERTDataSet(BaseDataSet):
    def get_max_len(self):
        maxlen = super().get_max_len()

        #  +2 for special tokens
        return maxlen + 2

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        labels = input_example.label
        sub_tokens = ['[CLS]']
        sub_token_indices = [-1]
        label_list = ['X']  # placeholder for special token
        label_mask = [0]  # value in (0, 1) - 0 signifies token to be masked

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [-1]

        # iterate over individual tokens and their labels
        for i in range(len(text)):
            token = text[i]
            label = labels[i]
            tokenized_token, num_sub_tokens = self.tokenize(token, i)

            for subtoken in tokenized_token:
                sub_tokens.append(subtoken)
                # record the original token index for each subtoken
                sub_token_indices.append(i)
                input_ids.append(
                    self.tokenizer.convert_tokens_to_ids(subtoken))

            if num_sub_tokens == 0:
                print(
                    f'WARNING: There is a word that BERT tokenizer \
                    returned nothing for: {token}')
                print('\t substituted the word with the unk token')
                sub_tokens.append(self.unk_token)
                sub_token_indices.append(-1)
                input_ids.append(
                    self.tokenizer.convert_tokens_to_ids(self.unk_token))

            label_list.append(label)
            label_mask.append(1)
            label_ids.append(self.label_map[label])

            # if num_sub_tokens > 1, mask the additional sub-tokens
            for i in range(1, num_sub_tokens):
                label_list.append(label)
                label_ids.append(self.label_map[label])
                label_mask.append(0)

        assert len(sub_token_indices) == len(sub_tokens) == len(label_list) ==\
            len(input_ids) == len(label_ids) == len(label_mask), \
            " ".join([str(len(sub_tokens)), str(
                len(label_list)), str(len(label_mask))])

        # truncate sentence if greater than max-len
        # additional -1 for the end of sentence token
        if len(sub_tokens) >= self.max_len:
            sub_token_indices = sub_token_indices[:(self.max_len - 1)]
            sub_tokens = sub_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        assert len(sub_tokens) < self.max_len, len(sub_tokens)

        # end of sentence tokens
        sub_tokens.append('[SEP]')
        sub_token_indices.append(-1)
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_list.append('X')
        label_ids.append(-1)
        label_mask.append(0)

        assert len(sub_token_indices) == len(sub_tokens) == \
            len(label_list) == len(input_ids) == len(
                label_ids) == len(label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        # padding
        while len(input_ids) < self.max_len:
            sub_token_indices.append(-1)
            sub_tokens.append('PAD')
            input_ids.append(
                self.tokenizer.convert_tokens_to_ids(self.pad_token))
            label_ids.append(-1)
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)
            label_list.append('C')

        assert len(sub_token_indices) == len(sub_tokens) == len(label_list) ==\
            len(input_ids) == len(label_ids) == len(attention_mask) == \
            len(sentence_id) == len(label_mask) == self.max_len, len(input_ids)

        return torch.LongTensor(sub_token_indices), sub_tokens, \
            label_list, torch.LongTensor(input_ids), \
            torch.LongTensor(label_ids), torch.LongTensor(attention_mask),\
            torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)
