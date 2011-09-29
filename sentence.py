class Sentence:
    def __init__(self, tokens):
        self._sen = list(tokens)

    def __len__(self):
        return len(self._sen)

    def __iter__(self):
        return iter(self._sen)
    def __reversed__(self):
        return reversed(self._sen)

    def __getitem__(self, key):
        return self._sen[key]

    def __setitem__(self, key, value):
        self._sen[key] = value

    def __delitem__(self, key):
        del self._sen[key]

    def __contains__(self, item):
        return item in self._index
    
    def __str__(self):
        return " ".join(self._sen)

    def ngram_positions(self, ngram):
        result = []

        for starter_index in (i for i, tok in enumerate(self._sen) if tok == ngram[0]):
            good = True
            for tok_i, tok in enumerate(ngram[1:]):
                try:
                    # search for remaining tokens
                    if self[starter_index + 1 + tok_i] == tok:
                        pass
                    else:
                        good = False
                        break
                except IndexError:
                    good = False
                    break
            if good:
                result.append(starter_index)
        return result

    def remove_ngram(self, ngram, backup=False):
        positions = self.ngram_positions(ngram)
        for pos in positions:
            if backup:
                if not "_backup_sen" in self.__dict__:
                    self._backup_sen = list(self._sen)

                for ngram_pos in xrange(len(ngram)):
                    self._backup_sen[pos+ngram_pos] = "[" + self._backup_sen[pos+ngram_pos] + "]"
            del self._sen[pos:pos+len(ngram)]

    def to_str(self, backup=False):
        return (" ".join(self._backup_sen) if hasattr(self, "_backup_sen") and backup else
                " ".join(self._sen))
