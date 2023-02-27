from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

from util import GEDLabel, ShuffleLabelStrategy as SLS, \
    SubwordFunctions
from util import from_file, format_error


class LightDataset(Dataset):

    def __init__(
        self, shuffle_labels, shuffle_strategy: SLS=SLS.ALL_TOKENS, labels2idx: dict=None, 
        ged_op: GEDLabel=GEDLabel.ALL, shuffle_seed: int=1, subword_function: SubwordFunctions=SubwordFunctions.LAST):

        self.shuffle_labels = shuffle_labels
        self.shuffle_labels_strategy = shuffle_strategy
        self.shuffle_seed = shuffle_seed
        self.labels2idx = labels2idx
        self.ged_op = ged_op
        self.subword_function = subword_function

        self.x = []
        self.y = []
        self.y_raw = []

        self.max_len = -1
        self.n_instances = 0
        self.is_data_loaded = False

    def get_max_len(self):
        print('ping')
        if not self.is_data_loaded:
            return -1

        for i in range(len(self.x)):
            assert len(self.x[i]) == len(self.y[i]) == len(self.y_mask[i])
            if len(self.x[i]) > self.max_len:
                self.max_len = len(self.x[i])

        return self.max_len

    def _shuffle_ungrammatical_labels_type_constrained(self, x_raw, y_raw):
        
        # create map from word type to set of errors
        #   e.g. 'the' = [R:DET, U:DET, M:DET]
        type2error = defaultdict(list)
        for sentence_index, _ in enumerate(y_raw):
            for token, label in zip(x_raw[sentence_index], y_raw[sentence_index]):
                if label != 'C':
                    type2error[token].append(label)
        
        num_errors_modified = 0  # for debugging
        num_errors = 0
        for i in range(len(y_raw)):
            for j in range(len(y_raw[i])):
                label = y_raw[i][j]
                if label != 'C':
                    token = x_raw[i][j]
                    
                    y_raw[i][j] = np.random.choice(type2error[token])
                    
                    num_errors += 1
                    if y_raw[i][j] != label:
                        num_errors_modified += 1
        
        return y_raw, num_errors_modified, num_errors

    def _shuffle_ungrammatical_labels(self, y_raw):
        # create set of error types
        #   e.g. errors = set([R:DET, U:DET, M:DET, ...])

        errors = set()
        for sentence_errors in y_raw:
            errors.update(sentence_errors)

        if 'C' in errors:  # remove 'correct' label from sampling set
            errors.remove('C')
        errors = list(errors)

        num_errors_modified = 0  # for debugging
        num_errors = 0
        for i in range(len(y_raw)):
            for j in range(len(y_raw[i])):
                label = y_raw[i][j]
                if label != 'C':
                    
                    y_raw[i][j] = np.random.choice(errors)
                    
                    num_errors += 1
                    if y_raw[i][j] != label:
                        num_errors_modified += 1
        
        return y_raw, num_errors_modified, num_errors

    def _shuffle_all_labels(self, y_raw):
        # create set of error types
        #   e.g. errors = set([C, R:DET, U:DET, M:DET, ...])
        errors = set()
        for sentence_errors in y_raw:
            errors.update(sentence_errors)
        errors = list(errors)
        
        num_errors_modified = 0  # for debugging
        num_errors = 0
        for i in range(len(y_raw)):
            for j in range(len(y_raw[i])):
                label = y_raw[i][j]
                
                y_raw[i][j] = np.random.choice(errors)
                
                num_errors += 1
                if y_raw[i][j] != label:
                    num_errors_modified += 1
        
        return y_raw, num_errors_modified, num_errors

    def _shuffle_labels(self, x_raw, y_raw):
        # option 1. shuffle just the tokens which have errors, where the set of errors are constrained to the type
        # option 2. shuffle just the tokens which have errors, sampling from the entire set of errors
        # option 2. shuffle all the labels (including correct tokens)

        # cast labels to <U12 to avoid a bug with <U10 character limit
        #   where it truncates R:VERB:TENSE to R:VERB:TEN
        for i in range(len(y_raw)):
            y_raw[i] = y_raw[i].astype('<U12')

        # set random seed for shuffling,
        #   but set it back afterwards for training randomness
        og_rng_state = np.random.get_state()
        np.random.seed(self.shuffle_seed)

        if self.shuffle_labels_strategy == SLS.CONSTRAINED_ERRORS_ONLY:
            y_raw, num_errors_modified, num_errors = self._shuffle_ungrammatical_labels_type_constrained(x_raw, y_raw)
        elif self.shuffle_labels_strategy == SLS.ERRORS_ONLY:
            y_raw, num_errors_modified, num_errors = self._shuffle_ungrammatical_labels(y_raw)
        elif self.shuffle_labels_strategy == SLS.ALL_TOKENS:
            y_raw, num_errors_modified, num_errors = self._shuffle_all_labels(y_raw)
            
        np.random.set_state(og_rng_state)

        print(f'Shuffled labels. {num_errors_modified} labels were changed out of {num_errors} in total ({num_errors_modified / float(num_errors)}%).')
        return y_raw

    def process_labels(self, x_raw, y_raw):

        y = []
        for sentence in y_raw:
            formatted_labels = [format_error(raw_label, self.ged_op) for raw_label in sentence]
            y.append(np.array(formatted_labels))

        # shuffle labels?
        if self.shuffle_labels:
            y = self._shuffle_labels(x_raw, y)

        return y

    def process_subword_units(self, x_raw, y_raw, y_mask, subword_indices):

        if self.subword_function == SubwordFunctions.FIRST:
            # loop through indices and create a boolean mask
            mask = []

            for sentence_indices in subword_indices:
                token_indices = set()
                sentence_mask = []
                for index in sentence_indices:
                    # check if we've already seen this index for this sentence
                    # e.g. for the indices: 0, 1, 2, 2, 2, 3, 4
                    # we would create a mask: T, T, T, F, F, T, T
                    if index not in token_indices:
                        token_indices.add(index)
                        sentence_mask.append(True)
                    else:
                        sentence_mask.append(False)
                mask.append(np.array(sentence_mask, dtype=bool))
            
            # mask = mask
            # use mask to reduce x, y
            x_raw = [embedding[m] for m, embedding in zip(mask, x_raw)]
            y_raw = [embedding[m] for m, embedding in zip(mask, y_raw)]
        elif self.subword_function == SubwordFunctions.LAST:
            # loop through indices backwards and create a boolean mask
            mask = []

            for sentence_indices in subword_indices:
                token_indices = set()
                sentence_mask = []
                for index in sentence_indices[::-1]:
                    # check if we've already seen this index for this sentence
                    # e.g. for the indices: 0, 1, 2, 2, 2, 3, 4
                    # we would create a mask: T, T, F, F, T, T, T
                    if index not in token_indices:
                        token_indices.add(index)
                        sentence_mask.append(True)
                    else:
                        sentence_mask.append(False)
                sentence_mask.reverse()
                mask.append(np.array(sentence_mask, dtype=bool))
            
            # use mask to reduce x, y
            x_raw = [embedding[m] for m, embedding in zip(mask, x_raw)]
            y_raw = [embedding[m] for m, embedding in zip(mask, y_raw)]
        elif self.subword_function == SubwordFunctions.SUM:
            for j, sentence_indices in enumerate(subword_indices):
                token_parts = {}
                
                x_sentence = []

                for i, index in enumerate(sentence_indices):
                    if index not in token_parts:
                        token_parts[index] = []
                    
                    token_parts[index].append(i)
                
                for token_index in sorted(list(token_parts.keys())):
                    summand_indices = token_parts[token_index]
                    x_slice = x_raw[j][summand_indices]
                    x_sum = np.sum(x_slice, axis=0)
                    x_sentence.append(
                        x_sum
                    )
                
                x_raw[j] = np.array(x_sentence)
                
            y_raw = [embedding[mask] for mask, embedding in zip(y_mask, y_raw)]

        return x_raw, y_raw, y_mask

    def process(self, x_raw, x, y_raw, y_mask, subword_indices):
        y_raw = self.process_labels(x_raw, y_raw)
        
        # process subword tokens
        x, y_raw, y_mask = self.process_subword_units(x, y_raw, y_mask, subword_indices)
        
        self.y_mask = y_mask
        self.subword_indices = subword_indices

        # convert from raw labels (R:VERB:SVA) to indices
        self.index_labels(y_raw)
        
        self.x = x
        self.x_raw = x_raw

        assert len(self.x) == len(self.y) == len(self.y_mask) == len(self.subword_indices)
        self.is_data_loaded = True

    def load_data(self, data_path: str, encoding_from_layer: int = None):
        x, x_raw, y_raw, y_mask, subword_indices = from_file(data_path, layer=encoding_from_layer)
        self.process(x_raw, x, y_raw, y_mask, subword_indices)

    def index_labels(self, y_raw):
        if self.labels2idx is None:
            y_labels = sorted(list(set([label for sentence in y_raw for label in sentence])))

            self.labels2idx = {label: idx for idx, label in enumerate(y_labels)}

        y = []
        for sentence in y_raw:
            indexed_labels = [self.labels2idx[raw_label] for raw_label in sentence]
            y.append(np.array(indexed_labels))
        self.y = np.array(y)
        self.y_raw = y_raw
        self.n_classes = len(self.labels2idx)

    def __len__(self):
        self.n_instances = len(self.x)
        return self.n_instances

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.y_mask[index])