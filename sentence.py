class Sentence:
    def __init__(self, tokens):
        self._sen = list(tokens)

    def __len__(self):
        return len(self._sen)

    def __iter__(self):
        return iter(self._sen)

    def __getitem__(self, key):
        return self._sen[key]

    def __setitem__(self, key, value):
        self._sen[key] = value

    def __delitem__(self, key):
        del self._sen[key]

    def __contains__(self, item):
        return item in self._sen
    
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

    def init_backup(self):
        if not "_backup_sen" in self.__dict__:
            self._backup_sen = list(self._sen)

    def remove_ngram(self, ngram, backup=False):
        positions = self.ngram_positions(ngram)
        if backup:
            if backup:
                self.init_backup()
            # look for position in backup sentence
            backup_positions = []
            for pos in positions:
                backup_pos = 0
                for i, tok in enumerate(self._backup_sen):
                    if type(tok) == tuple:
                        continue
                    else:
                        if backup_pos == pos:
                            backup_positions.append(i)
                            break
                        backup_pos += 1
         
        for i, pos in enumerate(positions):
            if backup:
                for ngram_pos in xrange(len(ngram)):
                    self._backup_sen[backup_positions[i]+ngram_pos] = (self._backup_sen[backup_positions[i]+ngram_pos],)
            shift = i * len(ngram)
            del self._sen[pos-shift:pos-shift+len(ngram)]

    def remove_toks(self, toks, backup=False):
        if backup:
            self.init_backup()
            for i, tok in enumerate(self._sen):
                if tok in toks:
                    self._backup_sen[i] = (tok,)
        self._sen = filter(lambda x: x not in toks, self._sen)

    def get_tokens(self, backup=False):
        if backup:
            self.init_backup()
            return self._backup_sen
        else:
            return self._sen
    
