"""
    Evaluate precise, recall and F1-score on sample-labeled-data.
"""
from pathlib import Path
from config import get_config
from solver import Solver
from data_loader import get_loader
from utils.load_labeled_data import load_data
data_path = Path('data')

def run():
    config = get_config()

    print(config)

    if config.checkpoint is None:
        print('Checkpoint is None!')
        return

    # read labeled data
    filename = data_path.joinpath('sample_labeled_data.txt')
    labeled_sequences = load_data(filename, spliter=',')

    solver = Solver(config)

    precise, recall, f1 = solver.eval(labeled_sequences)
    print(f'Precise is {precise}, Recall is {recall}, f1-Score is {f1}.')

if __name__ == '__main__':
    run()
