import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from train_probe import main as train_probe, create_arg_parser, parse_args
from src import ENCODING_PATH, SAVE_PROBE_PATH, TrainDatasetInfo, parse_minimal_args


def main(models: dict, datasets: dict):
    '''
        Iterate through datasets
        Train probes on TrainDatasetInfo's
        Use ENCODING_PATH to get saved encoded data
        Save probes to SAVE_PROBE_PATH
    '''

    # iterate through models
    for model_name, model_info in models.items():
        # iterate through datasets
        for dataset_name, dataset_info in datasets.items():
            if not type(dataset_info) == TrainDatasetInfo:
                continue

            encoded_data_path = os.path.join(ENCODING_PATH, dataset_name, model_name)

            dev_file_name = os.path.basename(dataset_info.dev_file)
            dev_folder_name = dataset_info.to_split_name(dev_file_name)

            # iterate through the training sets
            for training_file in dataset_info.train_files:

                # get filename and appropriate foldername
                train_file_name = os.path.basename(training_file)
                train_folder_name = dataset_info.to_split_name(train_file_name)

                for layer in range(1, 13):
                    # where to save the trained probe
                    save_name_parts = [
                        model_name,
                        dataset_name,
                        train_folder_name,
                        f'layer{layer}']

                    save_probe_path = os.path.join(
                        SAVE_PROBE_PATH,
                        '_'.join(save_name_parts)
                    )

                    parser = create_arg_parser()
                    str_args = [
                        "--encoded-data-path", encoded_data_path,
                        "--train-name", train_folder_name,
                        "--dev-name", dev_folder_name,
                        "--layer", str(layer),
                        "--model-name", model_name,
                        "--save-path", save_probe_path
                    ]

                    args = parser.parse_args(str_args)

                    train_probe(parse_args(args))


if __name__ == '__main__':
    models, datasets = parse_minimal_args()
    main(models, datasets)
