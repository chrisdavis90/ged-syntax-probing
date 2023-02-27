import json
import os
import glob
from torch.utils.data import DataLoader

from util import (num_labels_to_ged_label,
                ShuffleLabelStrategy as SLS, GEDLabel, SubwordFunctions)
from encoding.encode import load_data
from train import load_model, get_dataset, pad_batch, save_metrics
from encode_and_eval_probe import evaluate
from src import ENCODING_PATH, SAVE_PROBE_PATH, MODELS, DatasetInfo, get_datasets, create_minimal_args


def parse_layer_from_modelname(model_path: str) -> int:
    # check which layer we need to encode
    model_name = os.path.basename(os.path.dirname(model_path))
    model_name_parts = model_name.strip().split('_')
    layer_name = None
    for mnp in model_name_parts:
        if 'layer' in mnp:
            layer_name = mnp

    layer = None
    if layer_name is not None:
        layer = int(layer_name[5:])
    
    return layer


def read_model_subword_func(model_path: str) -> SubwordFunctions:
    # load saved args to get subword function
    model_saved_args_file = os.path.join(model_path, 'args.json')
    with open(model_saved_args_file, 'r') as f:
        model_args = json.load(f)

    subword_func = SubwordFunctions.FIRST
    if 'subword_func' in model_args:
        subword_func = SubwordFunctions[model_args['subword_func']]

    return subword_func


def load_probe(probe_type: str, model_saved_path: str):
    model, idx2labels = load_model(probe_type, model_saved_path)
    labels2idx = {l: i for i, l in idx2labels.items()}

    ged_op = num_labels_to_ged_label(len(idx2labels))
    print(f'Number of labels: {len(idx2labels)}')
    print(f'GED OP: {ged_op}')

    return model, labels2idx, ged_op


def create_dataset(
        labels2idx: dict, ged_op: GEDLabel, shuffle_labels: bool,
        shuffle_labels_strategy: SLS, subword_func: SubwordFunctions):
    
    dataset_cls = get_dataset()
    probe_dataset = dataset_cls(shuffle_labels, shuffle_labels_strategy, labels2idx, ged_op, 1, subword_func)

    return probe_dataset


def get_original_file(name: str, dataset: DatasetInfo):
    for file in dataset.data:
        filename = os.path.basename(file)
        if name in filename:
            return file

    return None


def save_predictions(target_file: str, samples: list, y_pred: list, y_probs: list):
    # 6. save predictions
    with open(target_file, 'w') as o2:
        for index in range(len(y_pred)):
            sentence_predictions = y_pred[index]
            sentence_probabilties = y_probs[index]
            tokens = samples[index].text
            goldlabels = samples[index].label
            assert len(tokens) == len(sentence_predictions) == len(sentence_probabilties), (len(tokens), len(sentence_predictions), len(sentence_probabilties))

            for word, goldlabel, predictedlabel, probabilities in zip(tokens, goldlabels, sentence_predictions, sentence_probabilties):
                line = [word, goldlabel, predictedlabel]
                line += list(probabilities.astype(str))
                o2.write('\t'.join(line) + '\n')
            o2.write('\n')


def evaluate_model_and_dataset(
        model_saved_path: str, encoded_eval_data_path: str,
        evaluation_dataset_name: str, save_results_folder: str,
        probe_type: str = 'linear', layer: int = None, batch_size: int = 32,
        num_workers: int = 4, shuffle_labels: bool = False,
        shuffle_labels_strategy: SLS = SLS.ALL_TOKENS,
        subword_func: SubwordFunctions = SubwordFunctions.LAST):

    if not os.path.exists(save_results_folder):
        os.makedirs(save_results_folder)

    # 1. load probe
    probe, labels2idx, ged_op = load_probe(probe_type, model_saved_path)

    # 2. create light dataset for the probe
    probe_dataset = create_dataset(labels2idx, ged_op, shuffle_labels, shuffle_labels_strategy, subword_func)

    # 3. iterate through encoded data
    # encoded_data_path is the top level folder, with one subfolder per
    #   evaluation file. Each subfolder should contain the encoded data
    #   for that evaluation file. I.e. the encoded representations..
    encoded_eval_folders = glob.glob(os.path.join(encoded_eval_data_path, '*'))

    # get datasets to retrieve original samples
    eval_dataset_info = get_datasets()[evaluation_dataset_name]
    dataset_reader = eval_dataset_info.dataset_reader

    for eval_folder in encoded_eval_folders:
        eval_name = os.path.basename(eval_folder)

        probe_dataset.load_data(data_path=eval_folder, encoding_from_layer=layer)

        data_iter = DataLoader(
            dataset=probe_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pad_batch)

        # 4. evaluate using loaded model
        metrics, y_pred, y_true, y_probs = evaluate(data_iter, probe, labels2idx)

        # 5. save metrics
        metrics_file = os.path.join(save_results_folder, f'{eval_name}_metrics.json')
        save_metrics(metrics_file, metrics)

        # load source data
        source_file_path = get_original_file(eval_name, eval_dataset_info)

        if source_file_path:
            samples, labels, tags2idx = load_data(source_file_path, dataset_reader)
            target_file = os.path.join(save_results_folder, f'gedPreds_{eval_name}.conll')

            save_predictions(target_file, samples, y_pred, y_probs)


def main(models, datasets):
    # iterate through models
    for model_name, model_info in models.items():
        saved_models = glob.glob(os.path.join(SAVE_PROBE_PATH, f'{model_name}*layer*', 'best_epoch'))
        for saved_model in saved_models:
            # iterate through datasets
            for dataset_name, dataset_info in datasets.items():
                encoded_data_path = os.path.join(ENCODING_PATH, dataset_name, model_name)
                save_results_folder = os.path.join(saved_model, dataset_name)

                subword_func = read_model_subword_func(saved_model)
                layer = parse_layer_from_modelname(saved_model)

                evaluate_model_and_dataset(
                    model_saved_path=saved_model,
                    encoded_eval_data_path=encoded_data_path,
                    evaluation_dataset_name=dataset_name,
                    save_results_folder=save_results_folder,
                    layer=layer,
                    subword_func=subword_func)


if __name__ == '__main__':
    models = MODELS
    datasets = get_datasets()

    # optionally specify a model and\or dataset to evaluate
    args = create_minimal_args().parse_args()

    # filter datasets
    if args.dataset:
        for dname in list(datasets.keys()):
            if args.dataset != dname:
                datasets.pop(dname, None)
    else:
        # drop training sets by default
        datasets.pop('wibea_rverbsva', None)
        datasets.pop('wiki_rverbsva', None)
        datasets.pop('wiki_rverbsva_notobe', None)

    # filter models
    if args.model:
        for mname in list(models.keys()):
            if args.model != mname:
                models.pop(mname, None)

    assert len(datasets) > 0
    assert len(models) > 0

    main(models, datasets)
