class Ngram:
    def __init__(self, *args):
        if len(args) == 1:
            self._ngram = tuple((args,))
        else:
            self._ngram = tuple(args)


    def __len__(self):
        return len(self._ngram)

    def __iter__(self):
        return iter(self._ngram)
    def __reversed__(self):
        return reversed(self._ngram)
    
    def __getitem__(self, key):
        return self._ngram[key]

    def __contains__(self, item):
        return item in self._ngram


