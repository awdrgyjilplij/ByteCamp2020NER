import sys
sys.path.append('../')
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from transformers import BertTokenizer
from utils.load_labeled_data import load_data
from utils.ahocorasick import AhoCorasick

data_path =  Path('../data')
pred_data_path = data_path.joinpath('pred_query_label.txt')
bert_path_or_name = '../prev_trained_model/chinese_wwm_ext_pytorch'
query_path = data_path.joinpath('byte_camp_query.txt')

tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)

def load_dict(filename):
    items = []
    with open(filename, encoding='utf8') as fin:
        for line in fin.readlines():
            line = line.strip('\n')
            items.append(line)

    return set(items)

def save_dict(filename, values):
    with open(filename, "w", encoding='utf8') as fout:
        for item in values:
            fout.write(item + '\n')
        print(f'Save to {str(filename)}, {len(values)} items.')

def read_query(filename, min_query_length, max_query_length):
    querys = []
    with open(filename, encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip('\n').strip()
            fileds = line.split('\t')

            if len(fileds) != 3:
                continue

            query = fileds[0]
            if len(query) < min_query_length or len(query) > max_query_length:
                continue
            querys.append(query)
    return querys

def get_freq_set(s, querys, topk):
    freq_dict = {}
    ac = AhoCorasick()
    for word in s:
        ac.addWord(word)
    ac.make()

    for query in tqdm(querys):
        result = ac.search(query)
        for l,r in result:
            word = query[l: r + 1]
            if word not in freq_dict:
                freq_dict[word] = 1
            else:
                freq_dict[word] += 1
    sorted_word = sorted(freq_dict.items(),key=lambda x:x[1], reverse=True)
    ret = set([k for k, v in sorted_word][:topk])
    return ret

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_dict_path', type=str, default='../data/dict')
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()
    org_dict_path = Path(args.org_dict_path)

    labeled_data = load_data(pred_data_path)


    district = set()
    business = set()
    house = set()
    developer = set()
    house_attr = set()

    def is_all_alpha(word):
        word = word.lower()
        flg = True
        for c in word:
            if not (c >= 'a' and c <= 'z'):
                flg = False
                break
        return flg

    for sequence, label in tqdm(labeled_data):
        assert len(sequence) == len(label)
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
                
                if len(word) < 3 or ('[UNK]' in word) or is_all_alpha(word):
                    continue
                if tag == 'district':
                    district.add(word)
                if tag == 'business':
                    business.add(word)
                if tag == 'house':
                    house.add(word)
                if tag == 'developer':
                    developer.add(word)
                if tag == 'house_attr':
                    house_attr.add(word)

    old_district = load_dict(org_dict_path.joinpath('district.txt'))
    old_business = load_dict(org_dict_path.joinpath('business.txt'))
    old_house = load_dict(org_dict_path.joinpath('house.txt'))
    old_developer = load_dict(org_dict_path.joinpath('developer.txt'))
    old_house_attr = load_dict(org_dict_path.joinpath('house_attr.txt'))

    district = district - old_district
    business = business - old_business
    house = house - old_house
    developer = developer - old_developer
    house_attr = house_attr - old_house_attr

    querys = read_query(query_path, 2, 25)

    district = get_freq_set(district, querys, args.topk) | old_district
    business = get_freq_set(business, querys, args.topk) | old_business
    house = get_freq_set(house, querys, args.topk) | old_house
    developer = get_freq_set(developer, querys, args.topk) | old_developer
    house_attr = get_freq_set(house_attr, querys, args.topk) | old_house_attr
    
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    new_dict_path = data_path.joinpath('dict').joinpath(time_now)
    new_dict_path.mkdir(exist_ok=True)
    
    save_dict(new_dict_path.joinpath('district.txt'), district)
    save_dict(new_dict_path.joinpath('business.txt'), business)
    save_dict(new_dict_path.joinpath('house.txt'), house)
    save_dict(new_dict_path.joinpath('developer.txt'), developer)
    save_dict(new_dict_path.joinpath('house_attr.txt'), house_attr)



if __name__ == '__main__':
    run()