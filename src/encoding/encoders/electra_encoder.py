import numpy as np
from torch.utils.data import DataLoader
from transformers import ElectraModel

from .base_encoder import BaseEncoder


class ElectraEncoder(BaseEncoder):

    def get_model(self, model_name_or_path):
        return ElectraModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
            output_attentions=True)

    def process(self, data_loader: DataLoader, only_last_layer=True):
        encoded_data = super().process(data_loader, only_last_layer)

        # remove special tokens
        if only_last_layer:
            encoded_data['vectors'] = np.array([x[1:-1] for x in encoded_data['vectors']])
        else:
            # for each data point
            for i, _ in enumerate(encoded_data['vectors']):
                # for each layer
                for j, _ in enumerate(encoded_data['vectors'][i]):
                    encoded_data['vectors'][i][j] = np.array(encoded_data['vectors'][i][j][1:-1])

        for name, vals in encoded_data['labels'].items():
            encoded_data['labels'][name] = np.array([x[1:-1] for x in vals])
        
        return encoded_data