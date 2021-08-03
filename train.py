from pathlib import Path
from config import get_config
from solver import Solver
from data_loader import get_loader
import argparse
from utils.load_labeled_data import load_data
data_path = Path('data')

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=None)
    config = get_config(parser)
    
    if config.file_name is None:
        print('No train file.')
        return

    # print config file
    print(config)
    config.save_path.mkdir(exist_ok=True)
    if config.checkpoint is None:
        with open(config.save_path.joinpath('config.txt'), 'w') as f:
            print(config, file=f)

    # read labeled data
    labeled_sequences = load_data(config.file_name)

    train_data_loader, val_data_loader = get_loader(labeled_sequences, batch_size=config.batch_size, val_batch_size=config.val_batch_size)

    solver = Solver(config, train_data_loader=train_data_loader, val_data_loader=val_data_loader)

    solver.train()



if __name__ == '__main__':
    run()
