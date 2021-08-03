import argparse
from pathlib import Path
from solver import Solver
from config import get_config
from utils.clear_text import clear_text

data_path = Path('data')
query_data_path = data_path.joinpath('byte_camp_query.txt')
pred_data_path = data_path.joinpath('pred_query_label.txt')

def get_unlabeled_data(filename, min_query_length, max_query_length):
    querys = []
    with open(filename) as fin:
        for line in fin.readlines():
            line = line.strip('\n').strip()
            fileds = line.split('\t')
            if len(fileds) != 3:
                continue

            query = clear_text(fileds[0])
            if len(query) > max_query_length or len(query) < min_query_length:
                continue
            querys.append(query)

    return querys

""" save the result of prediction """
def save_file(pred_data, filename):
    with open(filename, 'w') as fout:
        for query, label in pred_data:
            assert len(query) == len(label)
            for token, tag in zip(query, label):
                fout.write(token + '\t' + tag + '\n')
            fout.write('\n')
        print('Successfully Save prediction file.')


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_query_length', type=int, default=2)
    parser.add_argument('--max_query_length', type=int, default=25)

    config = get_config(parser)
        
    print(config)

    if config.checkpoint is None:
        print('Checkpoint is None!')
        return
    
    solver = Solver(config)

    querys = get_unlabeled_data(query_data_path, config.min_query_length, config.max_query_length)

    pred_data = solver.generate(querys)
    print(f'There is {len(pred_data)} pred data.')

    save_file(pred_data, pred_data_path)



if __name__ ==  '__main__':
    run()