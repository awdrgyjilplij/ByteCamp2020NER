import re
import sys
import random

sys.path.append('../')
import argparse
from pathlib import Path
from datetime import datetime
from transformers import BertTokenizer
from utils.clear_text import clear_text
from tqdm import tqdm

data_path = Path('../data')
labeled_data_dir = data_path.joinpath('labeled_data')
bert_path_or_name = '../prev_trained_model/chinese_wwm_ext_pytorch'
query_data_path = data_path.joinpath('byte_camp_query.txt')

tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
tag = {}  # 标注词典
numberreg = "([0-9]|(零|一|两|二|三|四|五|六|七|八|九|十|千|百))"  # 数字规则
matching = {
    'direction': "(朝|向|坐|朝向|面向|方向|面朝)" + "(北|东北|东|东南|南|西南|西|西北)" + "(方向|方)?",
    'unit_type': "(" + numberreg + "+|独|独立)" + "(室|房|卫|厅|厨)",
    'price': "(大约|大概)?" + numberreg + "+万" + numberreg + "*(块|元|元钱|块钱)?" +
             "(左右)?(每平米|每平方米|每平)?(每月|每天|每日|每年)?(以上|以下)?",
    'floor': "(地下)?(" + numberreg + "+|顶|底|低|高|中)(楼|层)(以上|以下)?",
    'area': "(大约|大概)?" + numberreg + "+(平米|平方|平方米|平)(以上|以下)?",
}

""" load dictionary, {set(), set(), ...} """


def load_dict(filename):
    items = []
    with open(filename, encoding='utf8') as fin:
        for line in fin.readlines():
            line = line.strip('\n')
            line = tokenizer.decode(tokenizer(line, add_special_tokens=False)['input_ids'])
            items.append(line)

    return set(items)


def is_in_dict(tokens, string):
    line = tokenizer.convert_tokens_to_string(tokens)
    if line in tag[string]:
        return True
    return False


def is_direction(tokens):
    line = tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
    if re.fullmatch(matching['direction'], line):
        return True
    return False


def is_unit_type(tokens):
    line = tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
    if re.fullmatch(matching['unit_type'], line):
        return True
    return False


def is_floor(tokens):
    line = tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
    if re.fullmatch(matching['floor'], line):
        return True
    return False


def is_price(tokens):
    line = tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
    if re.fullmatch(matching['price'], line):
        return True
    return False


def is_area(tokens):
    line = tokenizer.convert_tokens_to_string(tokens).replace(' ', '')
    if re.fullmatch(matching['area'], line):
        return True
    return False


def annonate(labels, lbound, rbound, tag_type):
    labels[lbound] = 'B-' + tag_type
    for i in range(lbound + 1, rbound):
        labels[i] = 'I-' + tag_type


"""forehead functionlist"""
functionlist = [
    (is_in_dict, 'district'),
    (is_in_dict, 'business'),
    (is_in_dict, 'house'),
    (is_in_dict, 'developer'),
    (is_in_dict, 'house_attr'),
    (is_direction, 'direction'),
    (is_unit_type, 'unit_type'),
    (is_area, 'area'),
    (is_price, 'price'),
    (is_floor, 'floor')
]

""" tokenize sequence and label tokens with dictionary"""


def label_sequence(line):
    input_ids = tokenizer(line)['input_ids']
    spt_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    n_token = len(spt_tokens)
    labels = ['O'] * n_token
    for length in range(n_token, 0, -1):
        for lbound in range(n_token - length):
            rbound = lbound + length  # [lbound, round)

            valid = True
            for i in range(lbound, rbound):  # 判断是否已经标注
                if labels[i] != 'O':
                    valid = False
                    break

            if not valid:
                continue

            part_sequence = spt_tokens[lbound: rbound]

            for (func, str) in functionlist:
                if str in tag:
                    if func(part_sequence, str):
                        annonate(labels, lbound, rbound, str)
                        break
                elif func(part_sequence):
                    annonate(labels, lbound, rbound, str)
                    break

    return (spt_tokens, labels)


""" remove those labeled data which only has O """


def clear_labeled_data(labeled_data):
    ret = []
    for spt_tokens, labels in labeled_data:
        if ''.join(labels) == 'O' * len(spt_tokens):
            continue
        ret.append((spt_tokens, labels))
    return ret


""" save file to a txt file """


def save_txt(labeled_data, filename):
    with open(filename, 'w', encoding='utf8') as fout:
        for spt_tokens, labels in labeled_data:
            for token, label in zip(spt_tokens, labels):
                fout.write(token + '\t' + label + '\n')
            fout.write('\n')


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_query_length', type=int, default=2)
    parser.add_argument('--max_query_length', type=int, default=25)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--dict_path', type=str, default='../data/dict')

    args = parser.parse_args()
    dict_path = Path(args.dict_path)
    for spliter in ['district', 'business', 'house', 'developer', 'house_attr']:
        tag[spliter] = load_dict(dict_path.joinpath(spliter + '.txt'))

    querys = []
    with open(query_data_path, encoding='utf8') as fin:
        for line in tqdm(fin.readlines(), desc="reading"):
            line = line.strip('\n')
            fileds = line.split('\t')

            if len(fileds) != 3:
                continue

            query = clear_text(fileds[0])
            if len(query) > args.max_query_length or len(query) < args.min_query_length:
                continue
            querys.append(query)

    querys = list(set(querys))
    print('sucessfully read query data.')

    if args.sample is None:

        labeled_data = [label_sequence(query) for query in tqdm(querys, desc="labeling")]
        labeled_data = clear_labeled_data(labeled_data)
        print(f'Sucessfully label data {len(labeled_data)}.')

        labeled_data_dir.mkdir(exist_ok=True)
        time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        labeled_data_path = labeled_data_dir.joinpath(time_now + '.txt')
        save_txt(labeled_data, labeled_data_path)
    else:
        random.shuffle(querys)
        print(args.sample)
        sample_querys = []
        for i in range(args.sample):
            sample_querys.append(querys[i])
        print(len(sample_querys))
        labeled_data = [label_sequence(query) for query in tqdm(sample_querys, desc="labeling")]

        print(f'Sucessfully label sample data {len(labeled_data)}.')
        save_txt(labeled_data, data_path.joinpath('sample_labeled_data.txt'))


if __name__ == '__main__':
    run()
