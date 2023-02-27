import json
import glob
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from .light_dataset import LightDataset
from .probes import LinearClassifier
from util import GEDLabel, ShuffleLabelStrategy as SLS, \
    SubwordFunctions


def pad_batch(batch):
        batch_size = len(batch)
        max_length = max([len(sentence[0]) for sentence in batch])
        shape = batch[0][0].shape[-1]

        x = torch.ones(batch_size, max_length, shape) * -1
        y = torch.ones(batch_size, max_length).long() * -1
        # y_mask = torch.zeros(batch_size, max_length).bool()

        for i, sentence in enumerate(batch):
            sent_len = len(sentence[0])
            x[i, :sent_len] = torch.from_numpy(sentence[0])
            y[i, :sent_len] = torch.from_numpy(sentence[1])
            # y_mask[i, :sent_len] = torch.from_numpy(sentence[2])

        if shape == 1:
            x = x.squeeze(-1).long()
            x[x == -1] = 0

        return (x, y)


def get_data_loader(dataset_cls: LightDataset, data_path: str, dataset_split, batch_size: int=32, shuffle_labels: bool=False, shuffle_labels_strategy: SLS=SLS.ALL_TOKENS, num_workers: int=1, ged_op: GEDLabel=GEDLabel.ALL, labels2idx: dict=None, shuffle_iterator: bool=False, encoding_from_layer: int=None, subword_function: SubwordFunctions=SubwordFunctions.FIRST, shuffle_seed=1):
    dataset = dataset_cls(shuffle_labels, shuffle_labels_strategy, labels2idx, ged_op, shuffle_seed, subword_function)
    dataset.load_data(os.path.join(data_path, dataset_split), encoding_from_layer)
    
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle_iterator,
                            num_workers=num_workers,
                            collate_fn=pad_batch)
    
    return dataset, data_loader


def get_data_loaders(dataset_cls: LightDataset, data_path: str, train_name: str, dev_name: str, batch_size: int=32, shuffle_labels: bool=False, shuffle_labels_strategy: SLS=SLS.ALL_TOKENS, num_workers: int=1, ged_op: GEDLabel=GEDLabel.ALL, subword_function: SubwordFunctions=SubwordFunctions.FIRST, encoding_from_layer: int=None):
    train_dataset, train_data_loader = get_data_loader(dataset_cls, data_path, train_name, batch_size, shuffle_labels, shuffle_labels_strategy, num_workers, ged_op, labels2idx=None, shuffle_iterator=True, encoding_from_layer=encoding_from_layer, subword_function=subword_function)
    
    _, dev_data_loader = get_data_loader(dataset_cls, data_path, dev_name, batch_size*2, False, shuffle_labels_strategy, num_workers, ged_op, train_dataset.labels2idx, shuffle_iterator=False, encoding_from_layer=encoding_from_layer, subword_function=subword_function)

    return train_data_loader, dev_data_loader, train_dataset.n_classes, train_dataset.labels2idx

def checkpoint(output_path, model, metrics, model_label, idx2labels, args):

    # save new epoch
    save_model(output_path, model, metrics, model_label, idx2labels, args)

    # save a copy of metrics to top level folder
    with open(os.path.join(output_path, f'{model_label}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # remove old epoch
    first, epoch_num = model_label.split('_')
    epoch_num = int(epoch_num)
    if epoch_num > 0:
        old_epoch_path = os.path.join(output_path, f'{first}_{epoch_num-1}')
        if os.path.exists(old_epoch_path):
            last_epoch_files = glob.glob(os.path.join(old_epoch_path, '*'))
            for f in last_epoch_files:
                os.remove(f)
            os.removedirs(old_epoch_path)


def get_dataset(model_name: str = None):
    return LightDataset


def get_model(probe_name: str):
    if probe_name == 'linear':
        return LinearClassifier


def save_metrics(target_file, metrics):
    with open(target_file, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_model(output_path, model, metrics, model_label, idx2labels, args):
    target_folder = os.path.join(output_path, model_label)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # save label dictionary
    np.save(os.path.join(target_folder, 'idx2labels.npy'), idx2labels)

    # save metrics
    save_metrics(os.path.join(target_folder, 'metrics.json'), metrics)

    # save script arguments
    args.ged_op = args.ged_op.name if type(args.ged_op) == GEDLabel else args.ged_op
    args.shuffle_labels_strategy = args.shuffle_labels_strategy.name \
        if type(args.shuffle_labels_strategy) == SLS else args.shuffle_labels_strategy
    args.subword_func = args.subword_func.name if type(args.subword_func) == SubwordFunctions \
        else args.subword_func

    with open(os.path.join(target_folder, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # save model
    fname = os.path.join(target_folder, 'model.pt')
    torch.save({
        'kwargs': model.get_args(),
        'model_state_dict': model.state_dict(),
    }, fname)


def load_model(probe_name, saved_path):
    # load model
    model_cls = get_model(probe_name)

    fname = os.path.join(saved_path, 'model.pt')
    checkpoint = torch.load(fname)
    
    model = model_cls(**checkpoint['kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # load label dictionary
    idx2labels = np.load(os.path.join(saved_path, 'idx2labels.npy'), allow_pickle=True).item()
    
    return model, idx2labels