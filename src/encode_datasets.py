import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoding import (create_dataset, get_encoder, get_pretrained_model_name,
                      load_data)
from src import ENCODING_PATH, get_encoded_data_path, parse_minimal_args
from util import to_file


def main(models: dict, datasets: dict):
    '''
        Load a pre-trained LM
        Iterate through datasets and save contextual representations to
            "ENCODING_PATH"
    '''

    num_workers = 4

    print('Encoding datasets')

    for model_name, model_info in models.items():        
        # issues batching with xlnet and the perm_mask
        batch_size = 1 if 'xlnet' in model_name else 64

        unidirectional = model_info.unidirectional

        # create encoder
        encoder_cls = get_encoder(encoder_name=model_name)
        encoder = encoder_cls(
            model_name_or_path=get_pretrained_model_name(model_name))

        # iterate through files in dataset
        for dataset_name, dataset_info in datasets.items():
            print(dataset_name)
            dataset_reader = dataset_info.dataset_reader

            for file_path in tqdm(dataset_info.data):
                # get filename and appropriate foldername
                filename = os.path.basename(file_path)
                data_folder_name = dataset_info.to_foldername(filename)
                # path to save encoded data
                output_path = get_encoded_data_path(
                    ENCODING_PATH,
                    dataset_name,
                    model_name,
                    data_folder_name
                )

                if os.path.exists(output_path):
                    print(f'Encoding for {filename} exists. Skipping.')
                    continue

                # load data
                samples, labels, tags2idx = load_data(file_path, dataset_reader)

                # create dataset
                dataset = create_dataset(
                    model_name,
                    samples,
                    tags2idx,
                    unidirectional)

                # create data loader to iterate in batches
                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)

                with torch.no_grad():
                    encoded_data = encoder.process(data_loader=data_loader, only_last_layer=False)

                # save
                to_file(encoded_data, output_path, only_last_layer=False)


if __name__ == "__main__":
    models, datasets = parse_minimal_args()
    main(models, datasets)
