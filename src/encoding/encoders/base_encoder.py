
from collections import defaultdict
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseEncoder(object):

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = self.get_model(model_name_or_path).to(DEVICE=DEVICE)
        self.model.eval()

    def get_model(self, model_name_or_path):
        pass

    def process_batch(self, batch, only_last_layer=True):

        b_sub_token_indices = batch[0].detach().cpu().numpy()  #np.array(batch[0]).transpose()
        b_subtokens = np.array(batch[1]).transpose()
        b_labels = np.array(batch[2]).transpose()
        batch = batch[3:]
        attn_mask_bool = batch[2].bool().numpy()
        
        # remove padding
        b_sub_token_indices = [sentence_subtokens[sentence_attn_mask] for sentence_subtokens, sentence_attn_mask in zip(b_sub_token_indices, attn_mask_bool)]
        b_subtokens = [sentence_subtokens[sentence_attn_mask] for sentence_subtokens, sentence_attn_mask in zip(b_subtokens, attn_mask_bool)]
        b_labels = [sentence_labels[sentence_attn_mask] for sentence_labels, sentence_attn_mask in zip(b_labels, attn_mask_bool)]

        batch = tuple(t.to(DEVICE) for t in batch)
        
        b_input_ids, b_label_ids, b_attention_mask, b_token_type_ids, b_label_masks = batch

        reps = self.embed_batch(input_ids=b_input_ids, attention_mask=b_attention_mask, token_type_ids=b_token_type_ids, only_last_layer=only_last_layer)

        b_label_masks = b_label_masks.detach().cpu().numpy()
        b_label_masks = [sentence_label_mask[sentence_attn_mask] for sentence_label_mask, sentence_attn_mask in zip(b_label_masks, attn_mask_bool)]

        return reps, b_subtokens, b_sub_token_indices, b_labels, b_label_masks


    def process(self, data_loader: DataLoader, only_last_layer=True):
        encoded_reps = []
        encoded_labels = defaultdict(list)

        for step, batch in enumerate(tqdm(data_loader)):
            
            reps, b_subtokens, b_sub_token_indices, b_labels, b_label_masks = self.process_batch(batch, only_last_layer)

            encoded_reps.extend(reps)
            encoded_labels['tokens'].extend(b_subtokens)
            encoded_labels['sub_token_indices'].extend(b_sub_token_indices)
            encoded_labels['labels'].extend(b_labels)
            encoded_labels['label_masks'].extend(b_label_masks)

        return {'vectors': encoded_reps, 'labels': encoded_labels}


    def embed_batch(self, input_ids, attention_mask, token_type_ids=None, only_last_layer=True):
        # encode
        with torch.no_grad():
            if token_type_ids is not None:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.model(input_ids=input_ids)

        # attentionmask = torch.BoolTensor(attention_mask)  #.to(DEVICE)
        attentionmask = attention_mask.bool()

        if only_last_layer:
            outputs = outputs[0]  # (b, max-len-of-texts, 768)

            # list of final layer representations
            sequence_outputs = []

            for batch_index in range(outputs.shape[0]):
                sequence_outputs.append(outputs[batch_index][attentionmask[batch_index]].detach().cpu().numpy())
        else:
            # outputs[layers][batch][max-len-of-texts][dim]
            # layer = 12
            # dim usually = 768 for BERT

            output_layers = [outputs['hidden_states'][i] for i in range(len(outputs['hidden_states']))]

            # list of lists
            # outer list is for inputs (batch size)
            # inner list is representations from layers (12 layers)
            # e.g. sequence_outputs[batch_size][layers][len-of-sentence][768]
            sequence_outputs = []
            
            for batch_i in range(output_layers[0].shape[0]):
                layer_outputs_for_example = []
                for layer_i in range(len(output_layers)):
                    layer_outputs_for_example.append(output_layers[layer_i][batch_i][attentionmask[batch_i]].detach().cpu().numpy())
                
                # assert len(layer_outputs_for_example) == 13  # 12 layers + input embeddings
                sequence_outputs.append(layer_outputs_for_example)

        return sequence_outputs
