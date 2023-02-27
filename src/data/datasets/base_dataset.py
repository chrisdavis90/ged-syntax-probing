from torch.utils import data
from util import create_tokenizer


class BaseDataSet(data.Dataset):
    def __init__(
            self, data_list: list, name: str, model_name_or_path: str,
            label_map: dict, max_len: int = None, unidirectional: bool = False):

        self.label_map = label_map
        self.data_list = data_list
        self.name = name
        self.model_name_or_path = model_name_or_path
        self.unidirectional = unidirectional
        
        self.tokenizer = create_tokenizer(
            model_name=model_name_or_path,
            saved_path_or_model_name=model_name_or_path
        )

        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        self.max_len = max_len
        if max_len is None:
            self.max_len = self.get_max_len()

    def __len__(self):
        return len(self.data_list)

    def tokenize(self, token: str, index: int):
        """
        Args:
            token (str): a single token to tokenize
            index (int): the index of the token in a sentence

        Returns:
            tuple: list of sub-word pieces (i.e. BPE, sentence piece, word piece)
                and the number of sub-word pieces
        """

        tokenized_token = self.tokenizer.tokenize(token)
        return tokenized_token, len(tokenized_token)

    def get_max_len(self):
        #  get maximum length of tokenized data
        maxlen = -1
        for datum in self.data_list:
            text = datum.text

            sentence_len = sum([self.tokenize(token, index)[1] for index, token in enumerate(text)])
            maxlen = sentence_len if sentence_len > maxlen else maxlen
        return maxlen

    def __getitem__(self, idx):
        pass