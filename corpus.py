from collections import defaultdict

from sentence import Sentence

class Corpus:
    def __init__(self, sentences=None):
        if sentences is not None:
            self._corpus = sentences
        else:
            self._corpus = []
        self.create_index()

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

    def add_sentence(self, sen):
        self._corpus.append(sen)
        sen_index = len(self._corpus)
        for tok in sen:
            self._index[tok].add(sen_index)

    def create_index(self):
        self._index = defaultdict(set)
        for i, sen in enumerate(self._corpus):
            for tok in sen:
                self._index[tok].add(i)
    
    def ngram_index(self, ngram):
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

    def remove_ngram(self, ngram):
        # TODO
        # indices counted twice
        for sen_i in self.ngram_index(ngram):
            sen = self._corpus[sen_i]
            sen.remove_ngram(ngram)

    @classmethod
    def read_from_file(f):
        c = Corpus()
        for l in f:
            le = l.strip().decode("utf-8").split()
            c.append(Sentence(le))

