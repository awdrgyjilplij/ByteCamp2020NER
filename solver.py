import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer
from utils.tag_dict import TokenDict
from utils.score import evaluate
from models.bert_for_ner import BertForNer

class Solver():

    def __init__(self, config, train_data_loader=None, val_data_loader=None, is_train=True):
        self.config = config
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.is_train = is_train
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path_or_name)
        self.tag_dict = TokenDict(self.config.tag_dict_path)
        self.build()

    def build(self):
        self.model = BertForNer(bert_path_or_name=self.config.bert_path_or_name, d_hidden=self.config.bert_hidden_size, 
                                n_tag=len(self.tag_dict)).to('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.checkpoint is not None:
           checkpoint = torch.load(self.config.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
           self.config.epoch_i = checkpoint['epoch']
           self.model.load_state_dict(checkpoint['model'])
           self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.decay_gamma)

    def save_model(self, epoch_i, accu):
        checkpoint = {'epoch': epoch_i, 'model': self.model.state_dict()}
        model_name = self.config.save_path.joinpath(str(accu))
        torch.save(checkpoint, model_name)

    """ Return Loss """
    def train_epoch(self, epoch_i, device = 'cuda' if torch.cuda.is_available() else 'cpu'
):
        self.model.train()

        loss_history = []
        n_word_total_history = []
        desc = '  - (Training)   '
        for batch_i, (src_sequences, tgt_sequences) in enumerate(tqdm(self.train_data_loader, desc=desc, ncols=80)):
            input_ids = [self.tokenizer.convert_tokens_to_ids(seq) for seq in src_sequences]
            attention_mask = [[idx != self.tokenizer.pad_token_id for idx in seq] for seq in input_ids]
            ground = [self.tag_dict.tokens_to_ids(seq) for seq in tgt_sequences]

            input_ids = torch.LongTensor(input_ids).to(device)
            attention_mask = torch.BoolTensor(attention_mask).to(device)
            labels = torch.tensor(ground).to(device)

            loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss_history.append(loss)
            n_word_total_history.append(attention_mask.sum().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if batch_i % self.config.print_every == 0:
                tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {sum(loss_history[-10:]) / sum(n_word_total_history[-10:]):.9f}')

        loss_per_word = sum(loss_history) / sum(n_word_total_history)
        return loss_per_word
    
    """ Return Accuracy """
    def eval_epoch(self, epoch_i, device = 'cuda' if torch.cuda.is_available() else 'cpu'
):
        self.model.eval()
        desc = '  - (Validation)   '
        
        precise_history = []
        recall_history = []
        f1_history = []

        with torch.no_grad():
            for batch_i, (src_sequences, tgt_sequences) in enumerate(tqdm(self.val_data_loader, desc=desc, ncols=80)):
                input_ids = [self.tokenizer.convert_tokens_to_ids(seq) for seq in src_sequences]
                attention_mask = [[idx != self.tokenizer.pad_token_id for idx in seq] for seq in input_ids]
                ground = [self.tag_dict.tokens_to_ids(seq) for seq in tgt_sequences]

                input_ids = torch.LongTensor(input_ids).to(device)
                attention_mask = torch.BoolTensor(attention_mask).to(device)
                labels = torch.tensor(ground).to(device)

                pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                
                if batch_i == 0:
                    batch_size = len(src_sequences)
                    for i in range(batch_size):
                        seq_len = input_ids[i].ne(self.tokenizer.pad_token_id).sum().item()
                        print(self.tokenizer.decode(input_ids[i].tolist()))
                        print(self.tag_dict.decode(pred[i].tolist()[:seq_len]))
                        print('==================================')

                # calculate precise, recall and f1 score
                true_seqs = self.tag_dict.ids_to_tokens(labels.reshape(-1).tolist())
                pred_seqs = self.tag_dict.ids_to_tokens(pred.reshape(-1).tolist())

                precise, recall, f1 = evaluate(true_seqs, pred_seqs, verbose=False)

                precise_history.append(precise)
                recall_history.append(recall)
                f1_history.append(f1)

                if batch_i % self.config.print_every == 0:
                    n_history = len(precise_history[-10:])
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: precise = {sum(precise_history[-10:]) / n_history:.9f}, recall = {sum(recall_history[-10:]) / n_history:.9f}, f1 score = {sum(f1_history[-10:]) / n_history:.9f}')

        n_batch = len(precise_history)
        precise_total = sum(precise_history) / n_batch
        recall_total = sum(recall_history) / n_batch
        f1_total = sum(f1_history) / n_batch

        return (precise_total, recall_total, f1_total)



    def train(self):
        
        for epoch_i in range(self.config.epoch_i, self.config.n_epoch):

            loss = self.train_epoch(epoch_i)

            precise, recall, f1 = self.eval_epoch(epoch_i)

            print(f'Precise is {precise:.9f}, Recall is {recall:.9f}, f1 Score is {f1:.9f}.')


            self.save_model(epoch_i, f1)

    """ Eval on labeled_data """
    def eval(self, labeled_data, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model.eval()
        sequences, labels = zip(*labeled_data)
        sequences = list(sequences)
        labels = list(labels)

        n_data = len(labeled_data)
        print(f'There is {n_data} query.')

        precise_history = []
        recall_history = []
        f1_history = []

        with torch.no_grad():
            for i in tqdm(range(0, n_data, self.config.batch_size)):
                src_sequences = sequences[i: min(i + self.config.batch_size, n_data)]
                tgt_sequences = labels[i: min(i + self.config.batch_size, n_data)]

                max_len = max([len(seq) for seq in src_sequences])
                input_ids = [self.tokenizer.convert_tokens_to_ids(seq) + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in src_sequences]
                attention_mask = [[idx != self.tokenizer.pad_token_id for idx in seq] for seq in input_ids]

                input_ids = torch.LongTensor(input_ids).to(device)
                attention_mask = torch.BoolTensor(attention_mask).to(device)


                pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                tgt_sequences = [seq + ['O'] * (max_len - len(seq)) for seq in tgt_sequences]
                true_seqs = [tag for seq in tgt_sequences for tag in seq]
                pred_seqs = self.tag_dict.ids_to_tokens(pred.reshape(-1).tolist())
                assert len(true_seqs) == len(pred_seqs)

                precise, recall, f1 = evaluate(true_seqs, pred_seqs, verbose=False)
                precise_history.append(precise)
                recall_history.append(recall)
                f1_history.append(f1)

        n_batch = len(precise_history)
        precise_total = sum(precise_history) / n_batch
        recall_total = sum(recall_history) / n_batch
        f1_total = sum(f1_history) / n_batch

        return (precise_total, recall_total, f1_total)


    def generate(self, unlabeled_data, device = 'cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        self.model.eval()
        random.shuffle(unlabeled_data)
        pred_data = []

        n_data = len(unlabeled_data)

        with torch.no_grad():
            for i in range(0, n_data, self.config.batch_size):
                sequences = unlabeled_data[i: min(i + self.config.batch_size, n_data)]
                inputs = self.tokenizer(sequences, padding=True)

                input_ids = torch.LongTensor(inputs['input_ids']).to(device)
                attention_mask = torch.BoolTensor(inputs['attention_mask']).to(device)

                pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)

                batch_size = len(sequences)
                seq_len = [attention_mask[i].sum().item() for i in range(batch_size)]
                tokens = [self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i][:seq_len[i]]) for i in range(batch_size)]
                pred_tag = [self.tag_dict.ids_to_tokens(pred[i].tolist()[:seq_len[i]]) for i in range(batch_size)]

                pred_data += zip(tokens, pred_tag)

                if i == 0 and verbose:
                    for sequence, predtag in pred_data :
                        assert len(sequence) == len(predtag)
                        print(' '.join(sequence))
                        print(' '.join(predtag))
                        print('==================================')

        return pred_data
