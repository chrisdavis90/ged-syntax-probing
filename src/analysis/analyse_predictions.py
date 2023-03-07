import glob
import os
from multiprocessing import Pool
from typing import Tuple

from sklearn.metrics import classification_report
from tqdm import tqdm


def marvinlinzen_verbs() -> list:
    return [
        'admire',
        'admires',
        'are',
        'is',
        'bring',
        'brings',
        'enjoy',
        'enjoys',
        'hate',
        'hates',
        'interest',
        'interests',
        'know',
        'knows',
        'laugh',
        'laughs',
        'like',
        'likes',
        'love',
        'loves',
        'smile',
        'smiles',
        'swim',
        'swims',
        'write',
        'writes',
    ]


def read_predictions(file: str, ignore_tobe_verbs: bool = False) -> list:
    mlverbs = set(marvinlinzen_verbs())
    if ignore_tobe_verbs:
        tobe_verbs = ['is', 'are']
        for i in tobe_verbs:
            mlverbs.remove(i)

    overlap = 0
    sentences_with = 0
    sentences_without = 0

    predictions = []
    sentences = 0
    totalsentences = 0
    with open(file, 'rt') as f:
        sentence = []
        sentence_contains_ignore_verb = False
        sentence_contains_ml_verb = False
        for line in f:
            if len(line.strip()) == 0:
                totalsentences += 1

                if not sentence_contains_ignore_verb:
                    predictions.extend(sentence)
                    sentences += 1
                    sentences_without += 1
                else:
                    if sentence_contains_ml_verb:
                        sentences += 1
                        for t in sentence:
                            if ignore_tobe_verbs:
                                # select only the tokens which aren't part of the ignore verbs
                                if t[0] not in tobe_verbs:
                                    predictions.append(t)
                            else:
                                predictions.append(t)
                    
                    sentences_with += 1
                
                if sentence_contains_ignore_verb and sentence_contains_ml_verb:
                    overlap += 1

                sentence = []
                sentence_contains_ignore_verb = False
                sentence_contains_ml_verb = False
                continue
        
            vals = line.strip().split()
            
            if ignore_tobe_verbs and vals[0] in tobe_verbs:
                sentence_contains_ignore_verb = True
            if vals[0] in mlverbs:
                sentence_contains_ml_verb = True

            sentence.append((vals[0], vals[1], vals[2]))
            # predictions.append((vals[1], vals[2]))

    new_predictions = []
    for p in predictions:
        new_predictions.append((p[1], p[2]))
    
    return new_predictions


def multiprocess(func, args, n_processes=6):
    with Pool(processes=n_processes) as pool:
        r = pool.map(func, args)

    return r


def binary_f1(y_true, y_pred) -> float:
    y_true = ['C' if x == 'C' else 'I' for x in y_true]
    y_true = ['C' if x == 'C' else 'I' for x in y_true]
    
    creport = classification_report(y_true, y_pred, output_dict=True, labels=['C','I'], zero_division=0)
    return creport['I']['f1-score']


def get_categories(category: str) -> list:

    if category == 'clause':
        return ['long_vp_coord', 'vp_coord', 'obj_rel_across', 'obj_rel_no_comp_across', 'obj_rel_within', 'obj_rel_no_comp_within', 'subj_rel', 'prep', 'simple', 'sent_comp']
    elif category == 'anim':
        return ['inanim', 'anim']


def group_by_category_per_layer(args) -> Tuple[int, dict]:
    layer, files, category = args

    categories = get_categories(category)

    preds = {}

    for c in categories:
        preds[c] = []

    for file in files:
        filename = os.path.basename(file)
        p = read_predictions(file)

        # logic to map filename to categories
        for c in categories:
            if c in filename:
                preds[c].extend(p)
                break
        
    # evaluate
    results = {}
    for group, predictions in preds.items():
        x, y = zip(*predictions)
        results[group] = binary_f1(x, y)
    
    return (layer, results)


def group_by_category(layer_predictions: dict, category: str) -> dict:

    layer_results = {}

    args = [(l, f, category) for l, f in layer_predictions.items()]

    res = multiprocess(group_by_category_per_layer, args)

    for r in res:
        l, r = r
        layer_results[l] = r

    return layer_results
        

def save_results(file_path: str, results: dict):

    sorted_layers = sorted([x for x in results.keys()])
    header = ['group'] + [str(x) for x in sorted_layers]
    # print(','.join(header))

    sorted_groups = sorted(list(results[sorted_layers[0]].keys()))

    with open(file_path, 'wt') as f:    
        f.write(','.join(header)+'\n')
        for group in sorted_groups:
            row = [group]

            for l in sorted_layers:
                row.append(results[l][group])
            
            row = [str(x) for x in row]
            f.write(','.join(row)+'\n')


def analyse_predictions(predictions: dict, results_file: str):
    output_folder = os.path.dirname(results_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_results(results_file, group_by_category(predictions, 'clause'))


if __name__ == '__main__':

    RESULTS_DIR = 'results'
    MODEL_DIR = '/path/to/saved/probes'

    model_names = ['bert', 'roberta', 'electra', 'gpt2', 'xlnet-uni', 'xlnet-bi']
    model_params = [
        ('wibea', 'train')
    ]

    # add wiki data
    for multiple in [1]:
        for version in range(1, 6):
            model_params.append(('wiki', f'rverbsva_train{multiple}xv{version}'))
            model_params.append(('wiki', f'rverbsva_nomlverbs_train{multiple}xv{version}'))

    for model_name in tqdm(model_names):
        for train_name, train_tag in tqdm(model_params):
            prediction_files = {}
            
            for layer in range(1, 13):
                pred_file_pattern = os.path.join(
                    MODEL_DIR,
                    f'{model_name}*{train_name}*{train_tag}*layer{layer}',
                    'best_epoch',
                    'marvin_linzen',
                    'gedPreds*')

                prediction_files[layer] = glob.glob(pred_file_pattern)

            results_file = os.path.join(
                RESULTS_DIR,
                f'{train_name}_{train_tag}',
                f'{model_name}_{train_name}_{train_tag}.txt')

            analyse_predictions(
                prediction_files,
                results_file
            )
