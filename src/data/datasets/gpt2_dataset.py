import torch
from data.datasets import BaseDataSet


class GPT2DataSet(BaseDataSet):
    def tokenize(self, token, index):
        #  add prefix space for all tokens except first token in the sentence
        if index == 0:
            tokenized_token = self.tokenizer.tokenize(token)
        else:
            tokenized_token = self.tokenizer.tokenize(token, add_prefix_space=True)
    
        return tokenized_token, len(tokenized_token)

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        labels = input_example.label
        subword_tokens = []
        sub_token_indices = []
        label_list = []
        label_mask = []  # value in (0, 1) - 0 signifies invalid token

        input_ids = []
        label_ids = []

        # iterate over individual tokens and their labels
        # for word, label in zip(text, label):
        for j in range(len(text)):
            token = text[j]
            label = labels[j]

            tokenized_token, num_sub_tokens = self.tokenize(token, j)

            for token in tokenized_token:
                subword_tokens.append(token)
                sub_token_indices.append(j)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            if len(tokenized_token) == 0:
                print("WARNING: There is a word that the GPT2 tokenizer returned nothing for:", token)
                print(f'\t substituted the word with a space')
                sub_token = ' '
                subword_tokens.append(sub_token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(sub_token))
            
            label_list.append(label)

            # assuming all labels in dataset have been seen (i.e. that there are no unknown labels)
            label_ids.append(self.label_map[label])
            label_mask.append(1)

            # if num_sub_tokens > 1, mask the additional sub-word-units
            for i in range(1, num_sub_tokens):
                label_list.append(label)
                label_ids.append(self.label_map[label])
                label_mask.append(0)
            
        assert len(sub_token_indices) == len(subword_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask), " ".join([str(len(subword_tokens)), str(len(label_list)), str(len(label_mask))])

        # truncate sentence if greater than max-len
        if len(subword_tokens) >= self.max_len:
            sub_token_indices = sub_token_indices[:(self.max_len)]
            subword_tokens = subword_tokens[:(self.max_len)]
            label_list = label_list[:(self.max_len)]
            input_ids = input_ids[:(self.max_len)]
            label_ids = label_ids[:(self.max_len)]
            label_mask = label_mask[:(self.max_len)]
            
        assert len(subword_tokens) <= self.max_len, len(subword_tokens)

        assert len(subword_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask) == len(sub_token_indices)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        # padding
        while len(input_ids) < self.max_len:
            sub_token_indices.append(-1)
            subword_tokens.append('PAD')
            input_ids.append(self.tokenizer.convert_tokens_to_ids('C'))
            label_ids.append(-1)
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)
            label_list.append('C')

        assert len(sub_token_indices) == len(subword_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)

        return torch.LongTensor(sub_token_indices), subword_tokens, label_list, torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(
            attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)