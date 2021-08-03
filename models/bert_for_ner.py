import torch
import torch.nn as nn
from transformers import BertModel
from .layers.crf import CRF

class BertForNer(nn.Module):
    """Bert for Ner.
    This module implements the combination of a Bert model and a CRF model.
    The forward computation of this class computes the loss of the given 
    input_ids and labels. This class also has a generate method which predicts
    the best labels of given input_ids.
    Args:
        bert_path_or_name: the path or name of Bert model.
        d_hidden: the dimension of Bert hidden layer.
        n_tag: the number of tags of CRF model.
        dropout: the ratio of Dropout bewteen Bert and classifier.
    Attributes:
        classifier (`~torch.nn.Linear`): A linear layer to transfer a d_hidden 
        dimension vector to a n_tag dimension vector.
        crf (`~CRF`): CRF model of num_tags of tags.
    """
    
    def __init__(self, bert_path_or_name, d_hidden, n_tag, dropout=0.1):
        super(BertForNer, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path_or_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_hidden, n_tag)
        self.crf = CRF(num_tags=n_tag, batch_first=True)
    """
    Input:
        input_ids: [batch_size, max_len]
        attention_mask: [batch_size, max_len]
        labels: [batch_size, max_len]
    Return:
        loss
    """
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        outputs = self.dropout(outputs)
        
        logits = self.classifier(outputs)
        loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)

        return -1 * loss
    """
    Input:
        input_ids: [batch_size, max_len]
        attention_mask: [batch_size, max_len]
    Return:
        prediction
    """
    def generate(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        outputs = self.dropout(outputs)

        logits = self.classifier(outputs)
        pred = self.crf.decode(emissions=logits, mask=attention_mask).squeeze(0)
        return pred

