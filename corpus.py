from collections import defaultdict

from sentence import Sentence

class Corpus:
    def __init__(self, backup, int_tokens=False):
        self._corpus = []
        self.create_index()

        self._backup = backup

        self._int_tokens = int_tokens
        if self._int_tokens:
            self._tokmap = {}

    def __len__(self):
        return len(self._corpus)

    def __iter__(self):
        return iter(self._corpus)
    def __reversed__(self):
        return reversed(self._corpus)

    def __getitem__(self, key):
        return self._corpus[key]

    def __setitem__(self, key, value):
        self._corpus[key] = value

    def __delitem__(self, key):
        del self._corpus[key]

    def append(self, item):
        self.add_sentence(item)

    def add_sentence(self, sen):
        # change tokens to ints
        if self._int_tokens:
            sen = self.tokens_to_ints(sen)

        # create actual Sentence instance
        new_sen = Sentence(sen)
        self._corpus.append(new_sen)

        # filter stopwords
        if hasattr(self, "_stopwords"):
            new_sen.remove_toks(self._stopwords, self._backup)

        # register to index
        sen_index = len(self._corpus) - 1
        for tok in new_sen:
            self._index[tok].add(sen_index)

    def create_index(self):
        self._index = defaultdict(set)
        for i, sen in enumerate(self._corpus):
            for tok in sen:
                self._index[tok].add(i)
    
    def ngram_index(self, ngram):
        if self._int_tokens:
            ngram = self.tokens_to_ints(ngram)
        if ngram[0] not in self._index:
            return set()

        occ = set(self._index[ngram[0]])
            
        if len(ngram) == 1:
            return occ

        for tok in ngram[1:]:
            occ = occ & self._index[tok]
                                                
        valid_occ = set()
        for sen_i in occ:
            sen = self._corpus[sen_i]
            if sen.ngram_positions(ngram):
                valid_occ.add(sen_i)
        return valid_occ

    def remove_ngram(self, ngram, ind=None, backup=False):
        ngram = self.tokens_to_ints(ngram)
        if ind is None:
            ind = self.ngram_index(ngram)
        for sen_i in ind:
            sen = self._corpus[sen_i]
            sen.remove_ngram(ngram, backup)

            # maintaining index
            for tok in ngram:
                if not tok in sen:
                    self._index[tok].remove(sen_i)
                    if len(self._index[tok]) == 0:
                        del self._index[tok]

    def tokens_to_ints(self, tokens):
        # sometimes tokens are already changed
        if type(tokens[0]) == int:
            return tokens

        ints = []
        for tok in tokens:
            if not tok in self._tokmap:
                self._tokmap[tok] = len(self._tokmap)
            ints.append(self._tokmap[tok])
        return ints

    def ints_to_tokens(self, ints):
        # first check, if there is a reverse dict
        if not hasattr(self, "_reverse_tokmap"):
            self._reverse_tokmap = dict((v,k) for k,v in self._tokmap.items())

        tokens = []
        for i in ints:
            # normal tokens
            if type(i) == int:
                tokens.append(self._reverse_tokmap[i])

            # removed tokens in backup mode
            else:
                tokens.append(u"[{0}]".format(self._reverse_tokmap[i[0]]))
        return tokens

    def set_stopwords(self, stopwords):
        if len(stopwords) != 0:
            self._stopwords = set(self.tokens_to_ints(list(stopwords)))

    def clean_multiple_hapax_sentences(self):
        # TODO implement if needed
        pass

    @staticmethod
    def read_from_file(f):
        c = Corpus()
        for l in f:
            le = l.strip().decode("utf-8").split()
            c.append(Sentence(le))

