import argparse
import os

import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm, trange

from train import (checkpoint, get_data_loaders, get_dataset, get_model,
                   load_model, save_model)
from util import GEDLabel
from util import ShuffleLabelStrategy as SLS
from util import SubwordFunctions as SubwordFuncs
from util import allowed_enum_items, config_seed, ged_evaluation

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--encoded-data-path", type=str, required=True, help='Path to folder containing train and dev representations')
    parser.add_argument("--train-name", type=str, required=False, help='Name of folder to use as train split', default='train')
    parser.add_argument("--dev-name", type=str, required=False, help='Name of folder to use as dev split', default='dev')
    parser.add_argument("--layer", type=int, required=False, help='Use the contextual representations extracted from a specific layer')
    parser.add_argument('--shuffle-labels', action='store_true')
    parser.add_argument('--shuffle-labels-strategy', type=str, required=False, help='How to shuffle labels', choices=allowed_enum_items(SLS))
    parser.add_argument('--batch-size', type=int, required=False, default=32)
    parser.add_argument('--ged-op', type=str, default='BINARY', choices=allowed_enum_items(GEDLabel))
    parser.add_argument('--subword-func', type=str, default='LAST', choices=allowed_enum_items(SubwordFuncs))
    # Model
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--probe', type=str, default='linear')
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embedding-size', type=int, default=768)
    # Optimization
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.0001)
    # Others
    parser.add_argument("--save-path", type=str, required=False, help='Path to folder to save model')
    parser.add_argument("--seed", type=int, default=100)

    return parser


def parse_args(args=None):
    if args is None:
        parser = create_arg_parser()
        args = parser.parse_args()
    
    args.ged_op = GEDLabel[args.ged_op]
    args.subword_func = SubwordFuncs[args.subword_func]

    # convert optional enum strings to enums
    if args.shuffle_labels_strategy:
        args.shuffle_labels_strategy = SLS[args.shuffle_labels_strategy]
    
    config_seed(args.seed)
    print(args)

    return args


def train(train_iter, dev_iter, model, epochs, patience, labels2idx, output_path, args):

    model = model.to(DEVICE)

    idx2labels = {idx: label for label, idx in labels2idx.items()}

    # training loop
    optimizer = optim.Adam(model.parameters(), weight_decay=args.alpha)

    best_metric = -1
    eps = 0.002

    epoch_iter = trange(0, epochs)
    for epoch_num in epoch_iter:
        model = model.train()
        tr_loss = 0
        nb_tr_steps = 0

        for batch in tqdm(train_iter):
            # x, y, y_mask = batch
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # y_mask = y_mask.to(DEVICE)

            optimizer.zero_grad()
            
            # loss, logits, labels = model(x, y, y_mask)
            loss, logits, labels = model(x, y)

            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1
            optimizer.step()

        train_loss = tr_loss / nb_tr_steps
        # print("Train loss: {}".format(train_loss))
        # logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))

        # evaluation loop
        model = model.eval()
        eval_loss, nb_eval_steps = 0, 0
        
        eval_predictions, eval_labels = [], []
        for batch in tqdm(dev_iter):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # y_mask = y_mask.to(DEVICE)

            with torch.no_grad():
                loss, logits, labels = model(x, y)

                probs = F.softmax(logits, dim=2).float()
                preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

                eval_predictions.extend(preds)
                eval_labels.extend(labels)

            eval_loss += loss.item()

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        # move to cpu
        eval_predictions = [y_pred.detach().cpu().numpy() for y_pred in eval_predictions]
        eval_labels = [y_true.detach().cpu().numpy() for y_true in eval_labels]

        # flatten
        y_pred = [p_i for p in eval_predictions for p_i in p]
        y_true = [l_i for l in eval_labels for l_i in l]
        
        assert len(y_pred) == len(y_true)

        # reduce to ignore padded items
        y_pred = [idx2labels[p] for p, t in zip(y_pred, y_true) if t != -1]
        y_true = [idx2labels[t] for t in y_true if t != -1]

        assert len(y_pred) == len(y_true)

        metrics = ged_evaluation(y_pred, y_true, verbose=False)
        metrics[f'epoch_{epoch_num}_train_loss'] = train_loss
        metrics[f'epoch_{epoch_num}_eval_loss'] = eval_loss

        epoch_iter.set_description(
            'Train loss: %.4f Dev loss: %.4f Binary F1: %.4f' % (train_loss, eval_loss, metrics['binarised_f1_I']))

        # set best
        eval_metric = metrics[f'binarised_f1_I']
        if eval_metric > best_metric + eps:
            best_metric = eval_metric
            best_epoch = epoch_num

            # save best model
            save_model(output_path, model, metrics, 'best_epoch', idx2labels, args)

        # Check-pointing to save the current epoch
        checkpoint(output_path, model, metrics, f'epoch_{epoch_num}', idx2labels, args)

        # check patience     
        if epoch_num - best_epoch >= patience:
            # end training early
            print(f'Patience reached. Ending training at epoch {epoch_num}')
            break

    model, _ = load_model('linear', os.path.join(output_path, 'best_epoch'))
    
    return model


def main(args):

    # 1. load data and create datasets, data loader
    dataset_cls = get_dataset()

    train_iter, dev_iter, n_classes, labels2idx = get_data_loaders(dataset_cls=dataset_cls,
                                            data_path=args.encoded_data_path,
                                            train_name=args.train_name,
                                            dev_name=args.dev_name,
                                            batch_size=args.batch_size,
                                            shuffle_labels=args.shuffle_labels,
                                            shuffle_labels_strategy=args.shuffle_labels_strategy,
                                            ged_op=args.ged_op,
                                            subword_function=args.subword_func,
                                            encoding_from_layer=args.layer)

    # 2. create model
    model_cls = get_model(args.probe)
    model = model_cls(hidden_size=args.embedding_size, num_labels=n_classes, dropout=args.dropout)

    # 3. train model
    model = train(train_iter, dev_iter, model, args.epochs, args.patience, labels2idx, args.save_path, args)


if __name__ == '__main__':
    main(parse_args())
