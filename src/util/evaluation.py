import logging
from sklearn.metrics import f1_score, fbeta_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from sklearn.metrics import confusion_matrix

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def ged_evaluation(y_pred, y_true, verbose=False):

    all_labels = sorted(list(set(y_true)))
    labels_excl_c = set(all_labels)
    labels_excl_c.remove('C')
    labels_excl_c = sorted(list(labels_excl_c))

    # y_true, y_pred
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    f05score_macro = precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='macro', zero_division=0)
    f05score_micro = precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='micro', zero_division=0)
    f05score_all = precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None, labels=all_labels, zero_division=0)
    f05score_weighted_excl_c = precision_recall_fscore_support(y_true, y_pred, beta=0.5, average='weighted', labels=labels_excl_c, zero_division=0)

    f1_macro_excl_c = precision_recall_fscore_support(y_true, y_pred, beta=1, average='macro', labels=labels_excl_c, zero_division=0)
    f1_micro_excl_c = precision_recall_fscore_support(y_true, y_pred, beta=1, average='micro', labels=labels_excl_c, zero_division=0)
    f1score_weighted_excl_c = precision_recall_fscore_support(y_true, y_pred, beta=1, average='weighted', labels=labels_excl_c, zero_division=0)

    creport = classification_report(y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0)

    binarised_y_true = ['I' if t!='C' else 'C' for t in y_true]
    binarised_y_pred = ['I' if t!='C' else 'C' for t in y_pred]
    binary_labels = ['C', 'I']
    creport_binarised = classification_report(binarised_y_true, binarised_y_pred, labels=binary_labels, output_dict=True, zero_division=0)
    binary_f05_score_all = fbeta_score(binarised_y_true, binarised_y_pred, beta=0.5, average=None, labels=binary_labels, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(binarised_y_true, binarised_y_pred, labels=binary_labels).ravel()

    precision_index = 0
    recall_index = 1
    fscore_index = 2
    metrics = {
        # f'{iter_data_name}_loss': eval_loss,
        'f1_macro': f1_macro,
        'f1_macro_excl_c': f1_macro_excl_c[fscore_index],
        'f1_precision_macro_excl_c': f1_macro_excl_c[precision_index],
        'f1_recall_macro_excl_c': f1_macro_excl_c[recall_index],
        'f1_micro_excl_c': f1_micro_excl_c[fscore_index],
        'f1_precision_micro_excl_c': f1_micro_excl_c[precision_index],
        'f1_recall_micro_excl_c': f1_micro_excl_c[recall_index],
        'f1_weighted': f1score_weighted_excl_c[fscore_index],
        'f1_precision_weighted': f1score_weighted_excl_c[precision_index],
        'f1_recall_weighted': f1score_weighted_excl_c[recall_index],
        'f1_precision_micro': f05score_micro[precision_index],
        'f05_precision_micro': f05score_micro[precision_index],
        'f05_recall_micro': f05score_micro[recall_index],
        'f05_macro': f05score_macro[fscore_index],
        'f05_precision_weighted': f05score_weighted_excl_c[precision_index],
        'f05_recall_weighted': f05score_weighted_excl_c[recall_index],
        'f05_weighted': f05score_weighted_excl_c[fscore_index],
        'binarised_f05_I': binary_f05_score_all[binary_labels.index('I')],
        'binarised_f1_I': creport_binarised['I']['f1-score'],
        'binarised_precision_I': creport_binarised['I']['precision'],
        'binarised_recall_I': creport_binarised['I']['recall'],
        'binarised_support_I': creport_binarised['I']['support'],
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item()
    }

    for idx, label in enumerate(all_labels):
        if label != 'X':
            metrics[f'f05_{label}'] = f05score_all[fscore_index][idx]
            metrics[f'precision_{label}'] = creport[label]['precision']
            metrics[f'recall_{label}'] = creport[label]['recall']
            metrics[f'f1_{label}'] = creport[label]['f1-score']
            metrics[f'support_{label}'] = creport[label]['support']

    if verbose:
        logger.info("Number of labels:", len(all_labels))
        logger.info("Macro averaged F0.5-Score: {}".format(f05score_macro))

        logger.info("\t P \t R \t F1 \t F0.5 \t support")
        for t in all_labels:
            if t != 'X':
                tag_metrics = "{}:\t{}\t{}\t{}\t{}\t{}".format(
                    t,
                    creport[t]['precision'],
                    creport[t]['recall'],
                    creport[t]['f1-score'],
                    f05score_all[all_labels.index(t)],
                    creport[t]['support'])
                logger.info(tag_metrics)

        logger.info("scores for binary GED (incorrect label): P:{}\t R:{}\t F0.5:{}\t Support:{}".format(
                creport_binarised['I']['precision'],
                creport_binarised['I']['recall'], 
                binary_f05_score_all[binary_labels.index('I')],
                creport_binarised['I']['support']))

    return metrics
