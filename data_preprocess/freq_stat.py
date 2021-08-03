"""
word frequencies statistic
"""
import jieba
from pathlib import Path
data_path = Path('../data')
stat_path = data_path.joinpath('stat')
query_data_path = data_path.joinpath('byte_camp_query.txt')

def run():
    freq_dict = {}
    with open(query_data_path) as fin:
        for line in fin.readlines():
            line = line.strip('\n').strip()
            fileds = line.split('\t')

            if len(fileds) != 3:
                continue
            
            spt_word = jieba.cut(fileds[0])
            for word in spt_word:
                if len(word) < 2:
                    continue
                if word in freq_dict:
                    freq_dict[word] = freq_dict[word] + 1
                else:
                    freq_dict[word] = 1
    
    sorted_word = sorted(freq_dict.items(),key=lambda x:x[1], reverse=True) # sort dict with frequency
    
    stat_path.mkdir(exist_ok=True)
    filename = stat_path.joinpath('word_freq.txt')
    with open(filename, 'w') as fout:
        for key, value in sorted_word:
            fout.write(key + ' ' + str(value) + '\n')


if __name__ == '__main__':
    run()