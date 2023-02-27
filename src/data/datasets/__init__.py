from torch.utils import data

from .base_dataset import BaseDataSet
from .bert_dataset import BERTDataSet
from .gpt2_dataset import GPT2DataSet
from .roberta_dataset import RobertaDataSet
from .xlnet_dataset import XLNetDataSet


def get_dataset(model_name: str) -> data.Dataset:
    if 'xlnet' in model_name:
        return XLNetDataSet
    elif model_name in ['roberta-base', 'roberta']:
        return RobertaDataSet
    elif model_name in ['bert-base', 'bert']:
        return BERTDataSet
    elif model_name == 'electra':
        return BERTDataSet
    elif model_name == 'gpt2':
        return GPT2DataSet
    else:
        return None
