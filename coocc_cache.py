from collections import defaultdict
import logging

class CooccCache:
    def __init__(self):
        self._cache = {}

    def add_sentence_pair(self, pair, index):
        src_sen, tgt_sen = pair
        for stok in src_sen:
            # create cache if needed
            try:
                self._cache[stok]
            except KeyError:
                self._cache[stok] = defaultdict(set)

            # add tokens to cache
            for ttok in tgt_sen:
                self._cache[stok][ttok].add(index)

    def possible_pairs(self, src_word, with_count=False):
        if with_count:
            return self._cache[src_word]
        else:
            return self._cache[src_word].keys()

    def coocc_count(self, pair):
        src, tgt = pair
        return len(self._cache[src][tgt])

    def filter(self, how_many=20):
        logging.info("Filtering coocc cache...")
        key = lambda x: len(x[1])
        import gc
        gc.disable()
        for src in self._cache:
            if len(self._cache[src]) > how_many:
                items = [i for i in self._cache[src].iteritems() if len(i[1]) > len(self._cache[src])/200]
            else:
                items = self._cache[src].items()
            bests = sorted(items, key=key, reverse=True)[:how_many]
            self._cache[src] = dict(bests)
        gc.enable()
        logging.info("Filtering coocc cache done.")

