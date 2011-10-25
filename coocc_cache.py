from collections import defaultdict
import logging

class CooccCache:
    def __init__(self, counts=True):
        self._cache = {}

        self._counts = counts

    def add_sentence_pair(self, pair, index):
        src_sen, tgt_sen = pair
        for stok in src_sen:
            # create cache if needed
            try:
                self._cache[stok]
            except KeyError:
                if self._counts:
                    self._cache[stok] = defaultdict(int)
                else:
                    self._cache[stok] = defaultdict(set)

            # add tokens to cache
            if self._counts:
                for ttok in set(tgt_sen):
                    self._cache[stok][ttok] += 1
            else:
                for ttok in tgt_sen:
                    self._cache[stok][ttok].add(index)

    def possible_pairs(self, src_word):
        if self._counts:
            return self._cache[src_word].keys()
        else:
            return self._cache[src_word].keys()

    def coocc_count(self, pair):
        src, tgt = pair
        if self._counts:
            return self._cache[src][tgt]
        else:
            return len(self._cache[src][tgt])

    def remove_pair(self, pair, indices=None):
        # if there is no occurence, remove nothing
        if indices is not None and len(indices) == 0:
            return
        
        src, tgt = pair
        if self._counts:
            if indices is None:
                del self._cache[src][tgt]
            else:
                self._cache[src][tgt] -= len(indices)
                if self._cache[src][tgt] == 0:
                    del self._cache[src][tgt]
                elif self._cache[src][tgt] < 0:
                    raise Exception("Error in cache, negative numbers")
        else:
            if indices is None:
                del self._cache[src][tgt]
            else:
                self._cache[src][tgt] -= indices
                if len(self._cache[src][tgt]) == 0:
                    del self._cache[src][tgt]

    def filter(self, how_many=20):
        logging.info("Filtering coocc cache...")
        if self._counts:
            key = lambda x: x[1]
        else:
            key = lambda x: len(x[1])
        for src in self._cache:
            if len(self._cache[src]) > 20:
                if self._counts:
                    items = filter(lambda x: x[1] > 1, self._cache[src].items())
                else:
                    items = filter(lambda x: len(x[1]) > 1, self._cache[src].items())
            else:
                items = self._cache[src].items()
            bests = sorted(items, key=key, reverse=True)[:how_many]
            self._cache[src] = dict(bests)
        logging.info("Filtering coocc cache done.")

