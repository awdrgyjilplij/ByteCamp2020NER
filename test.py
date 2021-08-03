from config import get_config
from solver import Solver
from transformers import BertTokenizer

bert_path_or_name = 'prev_trained_model/chinese_wwm_ext_pytorch'

tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)

def get_tag(sequence, label):
    ret = {}
    n_sequence = len(sequence)
    lb = 0
    while lb < n_sequence:
        if label[lb] == 'O':
            lb += 1
            continue
        else:
            rb = lb + 1
            tag = label[lb][2:]
            while rb < n_sequence and label[rb][2:] == tag:
                rb += 1

            word = tokenizer.convert_tokens_to_string(sequence[lb:rb]).replace(' ', '')
            lb = rb
            if tag not in ret:
                ret[tag] = word
            else:
                ret[tag] = [ret[teg]] + [word]
    return ret
            

def run():
    config = get_config()

    if config.checkpoint is None:
        print('Checkpoint is None!')
        return
    
    solver = Solver(config)

    while True:
        input_line = [input()]
        pred_data = solver.generate(input_line, verbose=False)
        sequence, label = pred_data[0]
        ans = get_tag(sequence, label)
        print(ans)

if __name__ == '__main__':
    run()
