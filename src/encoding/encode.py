import os
import sys
import argparse
import random
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data import DataProcessor
from encoding.encoders import (
    BertEncoder, ElectraEncoder, GPT2Encoder,
    RandomEncoder, RobertaEncoder, XLNetEncoder)
from data import DataProcessor
from data.datasets import get_dataset
from util import to_file


def get_encoder(encoder_name: str):
    if encoder_name in ['bert-base', 'bert']:
        return BertEncoder
    elif encoder_name == 'electra':
        return ElectraEncoder
    elif encoder_name in ['roberta-base', 'roberta']:
        return RobertaEncoder
    elif 'xlnet' in encoder_name:
        return XLNetEncoder
    elif encoder_name == 'gpt2':
        return GPT2Encoder
    elif encoder_name == 'random':
        return RandomEncoder
    else:
        print(f'{encoder_name} does not have a configured encoder.')
        exit()


def get_pretrained_model_name(model_name: str):
    if model_name in ['bert-base', 'bert']:
        return 'bert-base-cased'
    elif model_name == 'electra':
        return 'google/electra-base-discriminator'
    elif model_name in ['roberta-base', 'roberta']:
        return 'roberta-base'
    elif 'xlnet' in model_name:
        return 'xlnet-base-cased'
    elif model_name == 'gpt2':
        return 'gpt2'
    else:
        print(f'{model_name} does not have a configured pretrained model.')
        exit()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size",
                        help="The size of the mini batches",
                        default=1,
                        required=False,
                        type=int)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--unidirectional",
                        type=str,
                        required=False,
                        default='false',
                        help='Flag for XLNet: whether to use unidirectional decoding.')
    parser.add_argument("--only-last-layer",
                        required=False,
                        action='store_true',
                        help='Only encode the last layer')
    parser.add_argument("--path",
                        help="The path to dataset to encode",
                        required=True,
                        type=str)
    parser.add_argument("--dataset-reader",
                        help="The type of dataset reader",
                        required=False,
                        type=str,
                        default='default')
    parser.add_argument("--output-path",
                        help="The path to save the encoded representations",
                        required=True,
                        type=str)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--randomize-embeddings",
                       help="Randomly initialize embeddings",
                       required=False,
                       action='store_true')
    group.add_argument("--shuffle-tokens",
                       help="Shuffle input tokens before encoding",
                       required=False,
                       action='store_true')

    args = parser.parse_args()

    args.unidirectional = True if args.unidirectional.lower() == 'true' else False

    if args.model_name == 'xlnet':
        args.batch_size = 1

    print(args)

    args.num_workers = 1

    return args


def load_data(data_path: str, dataset_reader: str):
    # read data
    data_processor = DataProcessor(
        data_path,
        None,
        None,
        'ALL',
        dataset_reader
    )

    samples = data_processor.get_train_examples()
    labels = data_processor.get_labels()
    tags2idx = {}
    for i, label in enumerate(labels):
        tags2idx[label] = i

    return samples, labels, tags2idx


def shuffle_tokens(samples: list):
    all_tokens = []
    for sentence in samples:
        all_tokens.extend(sentence.text)

    random.shuffle(all_tokens)

    token_index = 0
    for i, _ in enumerate(samples):
        for j, _ in enumerate(samples[i].text):
            samples[i].text[j] = all_tokens[token_index]
            token_index += 1

    assert token_index == len(all_tokens)

    return samples


def create_dataset(
        model_name: str, samples: list, tags2idx: dict,
        unidirectional: bool = False) -> Dataset:

    dataset_class = get_dataset(model_name)

    dataset = dataset_class(
        data_list=samples,
        name=model_name,
        model_name_or_path=get_pretrained_model_name(model_name),
        label_map=tags2idx,
        max_len=None,
        unidirectional=unidirectional)

    return dataset


def process(
        encoder_name: str, model_name: str, dataset: Dataset,
        batch_size: int, only_last_layer: bool = True, num_workers: int = 4):

    # create data loader to iterate in batches
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    # create encoder
    encoder_cls = get_encoder(encoder_name=encoder_name)
    encoder = encoder_cls(
        model_name_or_path=get_pretrained_model_name(model_name))

    encoded_data = encoder.process(
        data_loader=data_loader, only_last_layer=only_last_layer)

    return encoded_data


def _main(
        data_path: str, dataset_reader: str, model_name: str, output_path: str,
        shuffle_tokens: bool = False, randomize_embeddings: bool = False,
        batch_size: int = 32, only_last_layer: bool = True,
        num_workers: int = 4, unidirectional: bool = False):

    samples, labels, tags2idx = load_data(data_path, dataset_reader)

    if shuffle_tokens:
        samples = shuffle_tokens(samples)

    dataset = create_dataset(model_name, samples, tags2idx, unidirectional)

    encoder_name = 'random' if randomize_embeddings else model_name

    with torch.no_grad():
        encoded_data = process(
            encoder_name=encoder_name, model_name=model_name,
            dataset=dataset, batch_size=batch_size,
            only_last_layer=only_last_layer, num_workers=num_workers)

    # save
    to_file(encoded_data, output_path, only_last_layer=only_last_layer)

    print("Process finished")


def main():
    args = get_args()

    _main(
        data_path=args.path, dataset_reader=args.dataset_reader,
        shuffle_tokens=args.shuffle_tokens, model_name=args.model_name,
        output_path=args.output_path,
        randomize_embeddings=args.randomize_embeddings,
        batch_size=args.batch_size, only_last_layer=args.only_last_layer,
        num_workers=args.num_workers, unidirectional=args.unidirectional)


if __name__ == "__main__":
    main()
