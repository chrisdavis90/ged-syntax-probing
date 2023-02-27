from util import GEDLabel
from util import readfile_multicolumn, readfile_singlecolumn
from .input_example import InputExample


class DataProcessor(object):
    """Processor for a GED data set in CONLL format"""

    def __init__(self, train_path: str, dev_path: str, test_path: str, ged_label: str, dataset_reader: str = 'default'):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.dataset_reader = dataset_reader
        self.ged_label = GEDLabel[ged_label]

    def get_train_examples(self):
        return self._create_examples(
            self._read_tsv(self.train_path, self.ged_label, self.dataset_reader), "train")

    def get_dev_examples(self):
        if self.dev_path != "":
            return self._create_examples(
                self._read_tsv(self.dev_path, self.ged_label, self.dataset_reader), "dev")
        else:
            return []

    def get_test_examples(self):
        return self._create_examples(
            self._read_tsv(self.test_path, self.ged_label, self.dataset_reader), "test")

    def get_labels(self):
        train = self._read_tsv(self.train_path, self.ged_label, self.dataset_reader)

        train_y = [[x for x in elem[1]] for elem in train]
        if self.dev_path is not None and len(self.dev_path) > 0:
            dev = self._read_tsv(self.dev_path, self.ged_label, self.dataset_reader)
        else:
            dev = []
        dev_y = [[x for x in elem[1]] for elem in dev]
        
        labels = sorted(list(
            set([elem for sublist in train_y + dev_y for elem in sublist])))
        
        return labels

    @staticmethod
    def _create_examples(lines, dataset_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            examples.append(InputExample(text=sentence, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file: str, ged_label: GEDLabel, dataset_reader: str):
        """Reads a tab separated value file."""
        if dataset_reader == 'default':
            return readfile_singlecolumn(input_file, ged_label)
        else:
            return readfile_multicolumn(input_file, ged_label)
