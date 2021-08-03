import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NerDataset(Dataset):
    """
    Args:
        labeled_sequences: list of tuple(src_seq, tgt_seq)
        src_seq: list of tokens
        tgt_seq: list of labels
    """
    def __init__(self, labeled_sequences):
        self.labeled_sequences = labeled_sequences
        self.len = len(self.labeled_sequences)

    """
    Return: tuple(src_seq, tgt_seq)
    """
    def __getitem__(self, index):
        return self.labeled_sequences[index]

    def __len__(self):
        return self.len


def get_loader(labeled_sequences, batch_size=64, val_batch_size=64, shuffle=True):


    def collate_fn(data):
        src_sequences, tgt_sequences = list(zip(*data))
        max_len = max([len(seq) for seq in src_sequences])
        src_sequences = [seq + ['[PAD]'] * (max_len - len(seq)) for seq in src_sequences]
        tgt_sequences = [seq + ['O'] * (max_len - len(seq)) for seq in tgt_sequences]

        return src_sequences, tgt_sequences

    dataset = NerDataset(labeled_sequences)

    # split dataset
    train_dataset_len = int(len(dataset) * 0.9)
    val_dataset_len = len(dataset) - train_dataset_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_len, val_dataset_len])
    
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return (train_data_loader, val_data_loader)