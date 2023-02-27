import torch
from data.datasets import BaseDataSet


class RobertaDataSet(BaseDataSet):

    def tokenize(self, token: str, index: int):
        """
        Args:
            token (str): a single token to tokenize
            index (int): the index of the token in a sentence

        Returns:
            tuple: list of sub-word pieces (i.e. BPE, sentence piece, word piece)
                and the number of sub-word pieces
        """
        
        # add prefix space for all tokens except first token in the sentence
        if index == 0:
            tokenized_token = self.tokenizer.tokenize(token)
        else:
            tokenized_token = self.tokenizer.tokenize(token, add_prefix_space=True)
    
        return tokenized_token, len(tokenized_token)

    def get_max_len(self):
        maxlen = super().get_max_len()
        return maxlen + 2  # +2 for special tokens

    def __getitem__(self, idx):

        '''
        From https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer

        Build model inputs from a sequence or a pair of
         sequence for sequence classification tasks by 
         concatenating and adding special tokens. A RoBERTa 
         sequence has the following format:

        single sequence: <s> X </s>
        pair of sequences: <s> A </s></s> B </s>
        '''

        input_example = self.data_list[idx]
        text = input_example.text
        labels = input_example.label
        sub_tokens = [self.tokenizer.cls_token]
        sub_token_indices = [-1]
        label_list = ['C']  # placeholder for special token, but it will be masked out
        label_mask = [0]  # value in (0, 1) - 0 signifies invalid token

        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]
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
                input_ids.append(self.tokenizer.convert_tokens_to_ids(subtoken))

            if num_sub_tokens == 0:
                print("WARNING: There is a word that BERT tokenizer returned nothin for:", token)
                print('\t substituted the word with "."')
                sub_tokens.append(self.unk_token)
                sub_token_indices.append(-1)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(self.unk_token))
            
            label_list.append(label)
            # some labels in the test data might be unseen and unknown to the training
            label_ids.append(self.label_map[label])
            label_mask.append(1)

            # if num_sub_tokens > 1, mask the additional sub-tokens
            for i in range(1, num_sub_tokens):
                label_list.append(label)
                label_ids.append(self.label_map[label])
                label_mask.append(0)

        # 
        # textsentence = ' '.join(text)
        # encoded_input = self.tokenizer(textsentence, return_tensors='pt')
        # for a, b in zip(encoded_input['input_ids'][0], input_ids):
        #     assert a == b

        assert len(sub_token_indices) == len(sub_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask), str(sub_tokens)+"*****"+ " ".join([str(len(sub_tokens)), str(len(label_list)), str(len(label_mask))])

        # truncate sentence if greater than max-len
        # additional -1 for the end of sentence token
        if len(sub_tokens) >= self.max_len:
            sub_token_indices = sub_token_indices[:(self.max_len - 1)]
            sub_tokens = sub_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            #tag_num = tag_num[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        assert len(sub_tokens) < self.max_len, len(sub_tokens)

        # end of sentence tokens
        sub_tokens.append(self.tokenizer.sep_token)
        sub_token_indices.append(-1)
        input_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
        label_list.append('C') # default to 'correct' label but this will be masked out
        label_ids.append(-1)
        label_mask.append(0)

        assert len(sub_token_indices) == len(sub_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        # padding
        while len(input_ids) < self.max_len:
            sub_token_indices.append(-1)
            sub_tokens.append(self.tokenizer.pad_token)
            input_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
            label_ids.append(-1)
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)
            label_list.append('C')

        assert len(sub_token_indices) == len(sub_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)

        return torch.LongTensor(sub_token_indices), sub_tokens, label_list, torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.LongTensor(attention_mask), torch.LongTensor(sentence_id), torch.BoolTensor(label_mask)