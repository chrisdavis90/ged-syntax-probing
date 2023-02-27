from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


class RandomEncoder(object):

    def __init__(self, model_name_or_path: str, data_loader: DataLoader):
        super().__init__()
        self.model = None
        self.data_loader = data_loader
        self.model_name_or_path = model_name_or_path
        self.dim = self.get_embedding_dim()

    def get_model(self, model_name_or_path):
        pass

    def get_embedding_dim(self):
        if 'bert-base' in self.model_name_or_path:
            return 768
        elif 'electra-base' in self.model_name_or_path:
            return 768
        elif self.model_name_or_path == 'gpt2':
            return 768
        

    def process(self, only_last_layer=True):
        encoded_reps = []
        encoded_labels = defaultdict(list)

        for step, batch in enumerate(tqdm(self.data_loader)):
            b_subtokens = np.array(batch[0]).transpose()
            b_labels = np.array(batch[1]).transpose()
            batch = batch[2:]
            attn_mask_bool = batch[2].bool().numpy()

            b_input_ids, b_label_ids, b_attention_mask, b_token_type_ids, b_label_masks = batch
            
            b_input_ids = b_input_ids.cpu().numpy()
            b_label_masks = b_label_masks.cpu().numpy()

            b_subtokens = [sentence_subtokens[sentence_attn_mask] for sentence_subtokens, sentence_attn_mask in zip(b_subtokens, attn_mask_bool)]
            b_labels = [sentence_labels[sentence_attn_mask] for sentence_labels, sentence_attn_mask in zip(b_labels, attn_mask_bool)]
            b_input_ids = [sentence_input_ids[sentence_attn_mask] for sentence_input_ids, sentence_attn_mask in zip(b_input_ids, attn_mask_bool)]
            b_label_masks = [sentence_label_mask[sentence_attn_mask] for sentence_label_mask, sentence_attn_mask in zip(b_label_masks, attn_mask_bool)]

            # create random embeddings between [-1/sqrt(dim), 1/sqrt(dim)]
            #   as per Weiting and Kiela (2019)
            # b_input_ids (b * sentence_len)
            reps = []
            low = float(-1)/np.sqrt(self.dim)
            high = float(1)/np.sqrt(self.dim)
            for batch_index in range(len(b_input_ids)):
                sentence_length = len(b_input_ids[batch_index])
                random_embedding = np.random.uniform(low=low, high=high, size=(sentence_length, self.dim))

                reps.append(random_embedding)
            
            encoded_reps.extend(np.array(reps))
            
            encoded_labels['tokens'].extend(b_subtokens)
            encoded_labels['labels'].extend(b_labels)
            encoded_labels['label_masks'].extend(b_label_masks)
        
        # remove special tokens
        encoded_reps = np.array([x[1:-1] for x in encoded_reps])

        for name, vals in encoded_labels.items():
            encoded_labels[name] = np.array([x[1:-1] for x in vals])

        return {'vectors': encoded_reps, 'labels': encoded_labels}
    