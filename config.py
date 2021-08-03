import argparse
import pprint
import torch
from datetime import datetime
from pathlib import Path
project_dir = Path(__file__).resolve().parent
save_dir = project_dir.joinpath('./ckpt/')

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        

        self.bert_path_or_name = 'prev_trained_model/chinese_wwm_ext_pytorch'
        self.tag_dict_path = 'data/dict/tag.txt'
        self.bert_hidden_size = 768 # bert pooler size

        # save path
        time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        save_dir.mkdir(exist_ok=True)
        self.save_path = save_dir.joinpath(time_now)
        if self.checkpoint is not None:
            self.save_path = Path(self.checkpoint).parent

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        self.epoch_i = 0



    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # save_setting
    parser.add_argument('--save_log', action='store_true')

    # load setting
    parser.add_argument('--checkpoint', type=str, default=None)
    
    # train setting
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--decay_gamma', type=float, default=0.5)
    
    parser.add_argument('--n_epoch', type=int, default=10) # epoch
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)

    # print setting
    parser.add_argument('--print_every', type=int, default=50)

    kwargs = parser.parse_args()
    
    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return Config(**kwargs)