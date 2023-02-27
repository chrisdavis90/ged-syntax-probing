import os

from edit_restrict_error_type_general import main as restrict
from m2_to_conll import main as m2_to_conll
from merge_conll_files import write_to_target as merge
from paths import PathInfo, data


def restrict_errors(file_info: PathInfo):
    restrict(
        m2_file=file_info.source,
        out_file=file_info.m2,
        restricted_error='R:VERB:SVA'
    )


def convert_to_conll(file_info: PathInfo):
    m2_to_conll(
        m2_file=file_info.m2,
        out=file_info.conll,
        all=True
    )


def create_target_folders(info: PathInfo):
    for target_file in [info.m2, info.conll]:
        folder = os.path.dirname(target_file)
        if not os.path.exists(folder):
            print(f'Creating directory: {folder}')
            os.makedirs(folder)


def process_learner_data(create_dirs=False):

    for _, info in data.items():
        # 0. Optionally create directories
        if create_dirs:
            create_target_folders(info)

        # 1. restrict errors to R:VERB:SVA
        restrict_errors(file_info=info)

        # 2. convert from .m2 to .conll
        convert_to_conll(file_info=info)

    # 3. combine wibea-train and fce-train
    #   use wibea target path
    target_folder = os.path.dirname(data['wibea-train'].conll)
    target_file = os.path.join(target_folder, 'wibeafce.train.gold.rverbsva.conll')
    merge(
        target_file=target_file,
        source_files=[
            data['wibea-train'].conll,
            data['fce-train'].conll
        ]
    )


if __name__ == '__main__':
    # learner data
    process_learner_data(create_dirs=True)
