import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from util import allowed_enum_items, ged_evaluation
from util import ShuffleLabelStrategy as SLS, num_labels_to_ged_label
from encoding import process as encode, load_data, create_dataset

from train import save_metrics, load_model, get_dataset, pad_batch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data-path", type=str, required=False,
        help='Path to dataset file')
    parser.add_argument("--dataset-reader", type=str, required=False,
        help='default or wibea')
    parser.add_argument('--shuffle-labels', action='store_true')
    parser.add_argument('--shuffle-labels-strategy', type=str, required=False,
        help='How to shuffle labels', choices=allowed_enum_items(SLS))
    parser.add_argument('--batch-size', type=int, default=1)
    # Model
    parser.add_argument('--model-path', type=str, help='Path to saved model')
    parser.add_argument('--model-name', type=str, default='bert-base')
    parser.add_argument('--probe', type=str, default='linear')
    # Others
    parser.add_argument("--output-path", type=str, required=False,
        help='Path to folder to save model')
    parser.add_argument("--include-probabilities", required=False,
        action='store_true', help='Flag to include predicted label probabilities')

    args = parser.parse_args()
    
    # check if which layer we need to encode
    model_name = os.path.basename(args.model_path)
    model_name_parts = model_name.strip().split('_')
    layer_name = None
    for mnp in model_name_parts:
        if 'layer' in mnp:
            layer_name = mnp
    
    if layer_name is not None:
        layer = int(layer_name[5:])
        args.only_last_layer = False
        args.layer = layer
    else:
        args.only_last_layer = True
        args.layer = -1

    args.num_workers = 4
    print(args)

    return args


def evaluate(data_iter, model, labels2idx):

    model = model.to(DEVICE)

    idx2labels = {idx: label for label, idx in labels2idx.items()}

    # evaluation loop
    model = model.eval()
    eval_loss, nb_eval_steps = 0, 0
    
    eval_predictions = []
    eval_labels = []
    eval_probabilities = []
    for batch in tqdm(data_iter):
        # x, y, y_mask = batch
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # y_mask = y_mask.to(DEVICE)

        with torch.no_grad():
            # loss, logits, labels = model(x, y, y_mask)
            loss, logits, labels = model(x, y)

            probs = F.softmax(logits, dim=2).float()
            preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            eval_probabilities.extend(probs)
            eval_predictions.extend(preds)
            eval_labels.extend(labels)

        eval_loss += loss.item()

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    # move to cpu
    eval_probabilities = [y_prob.detach().cpu().numpy() for y_prob in eval_probabilities]
    eval_predictions = [y_pred.detach().cpu().numpy() for y_pred in eval_predictions]
    eval_labels = [y_true.detach().cpu().numpy() for y_true in eval_labels]

    # flatten
    y_probs = [p_i for p in eval_probabilities for p_i in p]
    y_pred = [p_i for p in eval_predictions for p_i in p]
    y_true = [l_i for l in eval_labels for l_i in l]
    
    assert len(y_pred) == len(y_true) == len(y_probs)

    # reduce to ignore padded items
    y_probs = [p for p, t in zip(y_probs, y_true) if t != -1]
    y_pred = [idx2labels[p] for p, t in zip(y_pred, y_true) if t != -1]
    y_true = [idx2labels[t] for t in y_true if t != -1]

    assert len(y_pred) == len(y_true) == len(y_probs)

    metrics = ged_evaluation(y_pred, y_true, verbose=False)
    metrics['eval_loss'] = eval_loss
    
    # convert predictions to labels, maintaining sentence structure
    predictions = []
    gold = []
    probabilities = []
    for i in range(len(eval_predictions)):
        sentence_probabilities = [p for p, l in zip(eval_probabilities[i], eval_labels[i])  if l != -1]
        sentence_predictions = [idx2labels[p] for p, l in zip(eval_predictions[i], eval_labels[i])  if l != -1]
        sentence_labels = [idx2labels[l] for l in eval_labels[i] if l != -1]

        predictions.append(sentence_predictions)
        gold.append(sentence_labels)
        probabilities.append(sentence_probabilities)

    return metrics, predictions, gold, probabilities


def save_results(original_samples: list, y_pred: list, y_probs: list, metrics: dict, predictions_file: str, metrics_file: str):
    # save metrics
    save_metrics(metrics_file, metrics)

    # save predictions
    with open(predictions_file, 'w') as outfile:
        for sentence_info, sentence_predictions, sentence_probabilties in zip(original_samples, y_pred, y_probs):
            tokens = sentence_info.text
            goldlabels = sentence_info.label
            
            assert len(tokens) == len(sentence_predictions)

            for word, goldlabel, predictedlabel, probabilities in zip(tokens, goldlabels, sentence_predictions, sentence_probabilties):
                line = [word, goldlabel, predictedlabel]
                line += list(probabilities.astype(str))

                outfile.write('\t'.join(line) + '\n')

            outfile.write('\n')


def main():
    args = get_args()

    # load the processed dataset from .conll files
    samples, labels, tags2idx = load_data(args.data_path, args.dataset_reader)

    # create a dataset object for encoding using a pre-trained LM
    dataset = create_dataset(args.model_name, samples, tags2idx)

    # 1. encode the data
    encoded_data = encode(args.model_name, args.model_name, dataset, args.batch_size, args)

    # process encoded_data if we're using a specific layer
    if not args.only_last_layer:
        vectors = []

        for datum in encoded_data['vectors']:
            vectors.append(datum[args.layer])

        encoded_data['vectors'] = vectors

    # 2. load probe
    model, idx2labels = load_model(args.probe, args.model_path, 'best_epoch')

    args.ged_op = num_labels_to_ged_label(len(idx2labels))
    print(f'Number of labels: {len(idx2labels)}')
    print(f'GED OP: {args.ged_op}')

    # 3. create light-dataset object and data_loader for the probe
    dataset_cls = get_dataset(args.model_name)
    
    labels2idx = {l: i for i, l in idx2labels.items()}
    
    dataset = dataset_cls(args.shuffle_labels, args.shuffle_labels_strategy, labels2idx, args.ged_op)
    dataset.process(
        x=encoded_data['vectors'],
        x_raw=encoded_data['labels']['tokens'],
        y_raw=encoded_data['labels']['labels'],
        y_mask=encoded_data['labels']['label_masks']
    )

    data_iter = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=pad_batch)

    # 4. evaluate dataset using loaded model
    metrics, y_pred, y_true, y_probs = evaluate(data_iter, model, labels2idx)

    file_name = os.path.basename(args.data_path)
    ext_index = file_name.rindex('.')
    file_name = file_name[:ext_index]

    metrics_file = os.path.join(args.output_path, f'{file_name}_metrics.json')
    predictions_file = os.path.join(args.output_path, f'gedPreds_{file_name}.conll')

    save_results(
        samples, y_pred, y_probs, metrics, predictions_file, metrics_file)


if __name__ == '__main__':
    '''
    Encode and evaluate a dataset with a trained probe
    - differs from the other evaluation script because this encodes as well.
    '''
    main()
