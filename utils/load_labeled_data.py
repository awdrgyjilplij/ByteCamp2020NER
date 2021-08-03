""" 
    load labeled data 
    list of tuple(seqs, labels)
    seqs: list of tokens
    labels: list of label
"""
def load_data(filename, spliter='\t'):
    labeled_sequences = []
    with open(filename) as fin:
        seq = []
        label = []
        for line in fin.readlines():
            line = line.strip('\n').strip().strip(spliter)
            fileds = line.split(spliter)
            if line == '':
                assert len(seq) == len(label)
                if len(seq) > 0:
                    labeled_sequences.append((seq, label))
                seq = []
                label = []
            else:
                if len(fileds) != 2:
                    continue
                seq.append(fileds[0])
                label.append(fileds[1])
    
    if len(seq) > 0:
        assert len(seq) == len(label)
        labeled_sequences.append((seq, label))

    return labeled_sequences