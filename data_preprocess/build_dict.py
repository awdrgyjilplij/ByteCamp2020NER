"""
build dictionary for distant learning.
district  -- 城市/行政区
business  -- 商圈
house     -- 小区名/别称
developer -- 开发商
"""
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append('../')
from utils.clear_text import clear_text

data_path = Path('../data')
train_data_path = data_path.joinpath('byte_camp_house_info.txt')
extra_data_path = data_path.joinpath('loc.txt')
dict_path = data_path.joinpath('dict')

""" 保存词典 """


def save_dict(filename, values):
    with open(filename, "w", encoding='utf8') as fout:
        for item in values:
            fout.write(item + '\n')


def prune_district_dict(L):
    ret = []
    for item in L:
        if len(item) < 2:
            continue
        ret.append(item)
    return ret


def prune_business_dict(L):
    ret = []
    for item in L:
        if len(item) < 2:
            continue
        ret.append(item)
    return ret


def prune_house_dict(L):
    ret = []
    for item in L:
        if len(item) < 3:
            continue
        ret.append(item)
    return ret


def prune_developer_dict(L):
    ret = []
    for item in L:
        if len(item) < 2:
            continue
        ret.append(item)
    return ret


def run():
    district = []
    business = []
    house = []
    developer = []

    # dict from info
    with open(train_data_path, 'r', encoding='utf8') as fin:
        for line in tqdm(fin.readlines()):
            fileds = line.split('\t')
            if len(fileds) != 19:
                continue

            district.append(fileds[-6])
            district.append(fileds[-5])
            business.append(fileds[-4])
            house.append(fileds[-3])
            for val in fileds[-2]:
                house.append(val)
            developer.append(fileds[-1])

    # dict from loc
    with open(extra_data_path, 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            district.append(line)

    # remove duplication
    district = [clear_text(item) for item in list(set(district))]
    business = [clear_text(item) for item in list(set(business))]
    house = [clear_text(item) for item in list(set(house))]
    developer = [clear_text(item) for item in list(set(developer))]

    # prune dictionary
    district = prune_district_dict(district)
    business = prune_business_dict(business)
    house = prune_house_dict(house)
    developer = prune_developer_dict(developer)

    dict_path.mkdir(exist_ok=True)

    for tag, values in [('district', district), ('business', business), ('house', house),
                        ('developer', developer)]:
        filename = dict_path.joinpath(tag + '.txt')
        save_dict(filename, values)
        print(f'Save {tag} dictionary done.')


if __name__ == '__main__':
    run()
