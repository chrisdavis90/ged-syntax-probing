import os


def write_to_target(target_file, source_files):
    print(f'Writing to {os.path.basename(target_file)}')
    with open(target_file, 'wt') as outfile:
        for filepath in source_files:
            with open(filepath, 'r') as infile:
                for line in infile:
                    outfile.write(line)


def wibea_fce():
    write_to_target(
        target_file='data_simple/wibea/wibea_conll/wi+fce.train.rverbsva.conll',
        source_files=[
            'data_simple/wibea/wibea_conll/wi.train.gold.rverbsva.conll',
            'data_simple/fce/fce_conll/fce.train.gold.rverbsva.conll'
        ])


if __name__ == '__main__':
    wibea_fce()
