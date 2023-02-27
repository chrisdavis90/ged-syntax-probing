# coding=utf-8
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_prob = dropout

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.loss_func = CrossEntropyLoss(ignore_index=-1, reduction='mean')
    
    def get_args(self):
        return {
            'num_labels': self.num_labels,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout_prob
        }

    def forward(self, token_reps, labels=None, label_masks=None):

        sequence_output = self.dropout(token_reps)
        logits = self.classifier(sequence_output)  # (b, local_max_len, num_labels)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs + (labels,)
        
        return outputs
