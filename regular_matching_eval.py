"""
    Evaluate precise, recall and F1-score on sample-labeled-data.
"""
from pathlib import Path
from utils.load_labeled_data import load_data
from utils.score import evaluate

data_path = Path('data')


def run():
    # read labeled data
    filename = data_path.joinpath('pre_sample_labeled_data.txt')
    pre_labeled_sequences = load_data(filename, spliter='\t')
    filename = data_path.joinpath('sample_labeled_data.txt')
    labeled_sequences = load_data(filename, spliter=',')
    seqs, true_seqs = zip(*labeled_sequences)
    seqs, pred_seqs = zip(*pre_labeled_sequences)
    true_seqs = [token for seq in list(true_seqs) for token in seq]
    pred_seqs = [token for seq in list(pred_seqs) for token in seq]
    precise, recall, f1 = evaluate(true_seqs, pred_seqs, verbose=False)
    print(f'Precise is {precise}, Recall is {recall}, f1-Score is {f1}.')


if __name__ == '__main__':
    run()
