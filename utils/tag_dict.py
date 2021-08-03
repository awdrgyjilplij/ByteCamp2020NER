""" transform token to id, or transform id to token """

def get_tokens(filename):
    ret = []
    with open(filename) as fin:
        for line in fin.readlines():
            line = line.strip('\t').strip()
            ret.append(line)
    return ret

class TokenDict():
    def __init__(self, filename):
        self.id2token = {}
        self.token2id = {}

        tokens = get_tokens(filename)
        for i, token in enumerate(tokens):
            self.token2id[token] = i
            self.id2token[i] = token

    def __len__(self):
        return len(self.token2id)

    def token_to_id(self, token):
        return self.token2id[token]
    
    def id_to_token(self, nid):
        return self.id2token[nid]
    
    def tokens_to_ids(self, tokens):
        return [self.token_to_id(token) for token in tokens]
    
    def ids_to_tokens(self, nids):
        return [self.id_to_token(nid) for nid in nids]
    
    def decode(self, nids):
        L = self.ids_to_tokens(nids)
        return ' '.join(L)