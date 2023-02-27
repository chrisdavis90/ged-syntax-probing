from transformers import GPT2Model

from .base_encoder import BaseEncoder


class GPT2Encoder(BaseEncoder):

    def get_model(self, model_name_or_path):
        return GPT2Model.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
            output_attentions=True)
