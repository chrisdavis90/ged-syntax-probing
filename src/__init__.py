from .data import *
from .util import *
from .encoding import *
from .train import *
from .analysis import *

# where to save encoded data (contextual representations)
ENCODING_PATH = '/path/to/encoding/folder'
# where to save the trained probes
SAVE_PROBE_PATH = '/path/to/save/folder'

# update these paths if you've moved the data
MARVIN_LINZEN_BASE = 'datasets/marvin_linzen/stimuli'
WIKEDITS_BASE = 'datasets/wikedits'
WIBEA_BASE = 'datasets/wibea/wibea_conll'


class ModelInfo(object):
    def __init__(self, model, unidirectional) -> None:
        self.model = model
        self.unidirectional = unidirectional


MODELS = {
    'bert': ModelInfo('bert-base', False),
    'gpt2': ModelInfo('gpt2', False),
    'electra': ModelInfo('electra', False),
    'roberta': ModelInfo('roberta-base', False),
    'xlnet-uni': ModelInfo('xlnet-uni', True),
    'xlnet-bi': ModelInfo('xlnet-bi', False)
}


class DatasetInfo(object):
    def __init__(self, basepath, reader, data) -> None:
        self.basepath = basepath
        self.dataset_reader = reader
        self.data = data

    def to_foldername(self, filename: str) -> str:
        ext_index = filename.rindex('.')
        return filename[:ext_index]


class TrainDatasetInfo(DatasetInfo):
    def __init__(self, basepath, reader, data) -> None:
        super().__init__(basepath, reader, data)

        self.train_files = []
        self.dev_file = None
        
        # group into train and dev
        for file in data:
            filename = os.path.basename(file)
            if 'dev' in filename:
                self.dev_file = file
            else:
                self.train_files.append(file)

    def to_foldername(self, filename: str) -> str:
        return self.to_split_name(filename)

    def to_split_name(self, filename: str) -> str:
        # convert filename to either train or dev
        parts = filename.strip().split('.')

        if 'wibea' in filename:
            return parts[1]
        elif 'wiked' in filename:
            if 'dev' in filename:
                split_name = 'dev'
            else:
                split = parts[-3]
                version = parts[-2]
                split_name = f'{split}{version}'

            return split_name


def get_encoded_data_path(base_path: str, dataset_name: str, model_name: str, data_folder_name: str) -> str:
    return os.path.join(
        base_path,
        dataset_name,
        model_name,
        data_folder_name
    )


def get_wikiedits_files(without_ml_verbs: bool = False) -> list:
    if without_ml_verbs:
        file_pattern = 'wiked.tok.r-verb-sva.nomlverbs.train*.conll'
        dev_file_pattern = 'wiked.tok.r-verb-sva.nomlverbs.dev.conll'
    else:
        file_pattern = 'wiked.tok.r-verb-sva.train.conll'
        dev_file_pattern = 'wiked.tok.r-verb-sva.train.conll'

    files = glob.glob(os.path.join(WIKEDITS_BASE, file_pattern))
    files.append(os.path.join(WIKEDITS_BASE, dev_file_pattern))
    return files


def get_datasets() -> dict:
    # list of files to evaluate
    datasets = {
        'marvin_linzen': DatasetInfo(
            basepath=MARVIN_LINZEN_BASE,
            reader='default',
            data=glob.glob(os.path.join(MARVIN_LINZEN_BASE, '*.conll'))
        ),
        'wibea_rverbsva': TrainDatasetInfo(
            basepath=WIBEA_BASE,
            reader='default',
            data=[
                os.path.join(WIBEA_BASE, 'wibeafce.train.gold.rverbsva.conll'),
                os.path.join(WIBEA_BASE, 'wibea.dev.gold.rverbsva.conll')
            ]
        ),
        'wiki_rverbsva': TrainDatasetInfo(
            basepath=WIKEDITS_BASE,
            reader='default',
            data=get_wikiedits_files(without_ml_verbs=False)
        ),
        'wiki_rverbsva_nomlverbs': TrainDatasetInfo(
            basepath=WIKEDITS_BASE,
            reader='default',
            data=get_wikiedits_files(without_ml_verbs=True)
        )
    }

    return datasets


def create_minimal_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=False, help='Name of a model', choices=list(MODELS.keys()))
    parser.add_argument("-d", "--dataset", type=str, required=False, help='Name of dataset', choices=list(get_datasets().keys()))
    return parser


def parse_minimal_args(args=None):
    '''
        Returns all models and datasets unless a specific
            model and/or dataset is specified through the args
    '''
    if args is None:
        args = create_minimal_args().parse_args()

    models = MODELS
    datasets = get_datasets()

    # filter datasets
    if args.dataset:
        for dname in list(datasets.keys()):
            if args.dataset != dname:
                datasets.pop(dname, None)

    # filter models
    if args.model:
        for mname in list(models.keys()):
            if args.model != mname:
                models.pop(mname, None)

    assert len(datasets) > 0
    assert len(models) > 0

    return models, datasets
