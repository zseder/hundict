import logging
import gc
from collections import defaultdict
from operator import itemgetter

from corpus import Corpus

class BiCorpus:
    def __init__(self, backup=False, tok_coocc_caching=True, int_tokens=False):
        self._src = Corpus(backup, int_tokens)
        self._tgt = Corpus(backup, int_tokens)
        self._backup = backup
        self._coocc_caching = tok_coocc_caching
        if self._coocc_caching:
            self._src_coocc_cache = {}

    def write(self, out):
        for sen_i in xrange(len(self._src)):
            src_sen, tgt_sen = self._src[sen_i], self._tgt[sen_i]
            src_str = " ".join(self._src.ints_to_tokens(src_sen.get_tokens(self._backup)))
            tgt_str = " ".join(self._tgt.ints_to_tokens(tgt_sen.get_tokens(self._backup)))
            out.write(u"{0}\t{1}\n".format(src_str, tgt_str).encode("utf-8"))

    def add_sentence_pair(self, pair):
        src, tgt = pair
        self._src.append(src)
        self._tgt.append(tgt)

        # cache cooccurences
        if self._coocc_caching:
            for stok in self._src[-1]:
                # create cache if needed
                try:
                    self._src_coocc_cache[stok]
                except KeyError:
                    self._src_coocc_cache[stok] = (defaultdict(int), 0)

                # add tokens to cache
                for ttok in self._tgt[-1]:
                    if (ttok in self._src_coocc_cache[stok][0] or
                        self._src_coocc_cache[stok][1] <= 10):
                        self._src_coocc_cache[stok][0][ttok] += 1
                        if self._src_coocc_cache[stok][0][ttok] > self._src_coocc_cache[stok][1]:
                            self._src_coocc_cache[stok] = (self._src_coocc_cache[stok][0],self._src_coocc_cache[stok][0][ttok])

    def ngram_pair_context(self, pair, max_len=None):
        src, tgt = pair
        def __insert_contexts(occ, insterter):
            for sen_index in occ:
                src_sen = self._src[sen_index]
                tgt_sen = self._tgt[sen_index]
                if max_len is None:
                    insterter((src_sen, tgt_sen))
                else:
                    src_ngram_indices = src_sen.ngram_positions(src)
                    tgt_ngram_indices = tgt_sen.ngram_positions(tgt)
                    for src_ngram_index in src_ngram_indices:
                        src_left = max(0, src_ngram_index - max_len)
                        src_right = min(len(src_sen), src_ngram_index + len(src) + max_len)
                        for tgt_ngram_index in tgt_ngram_indices:
                            tgt_left = max(0, tgt_ngram_index - max_len)
                            tgt_right = min(len(tgt_sen), tgt_ngram_index + len(tgt) + max_len)
                            insterter((
                                (src_sen[src_left:src_ngram_index],src_sen[src_ngram_index + 1:src_right]),
                                (tgt_sen[tgt_left:tgt_ngram_index],tgt_sen[tgt_ngram_index + 1:tgt_right])
                                ))

        src_occ = self._src.ngram_index(src)
        tgt_occ = self._tgt.ngram_index(tgt)
        coocc = src_occ & tgt_occ

        context = [], [], []
        __insert_contexts(coocc, context[0].append)
        __insert_contexts(src_occ - coocc, context[1].append)
        __insert_contexts(tgt_occ - coocc, context[2].append)
        
        return context

    def remove_ngram_pair(self, pair):
        src, tgt = pair
        indices = self._src.ngram_index(src) & self._tgt.ngram_index(tgt)
        self._src.remove_ngram(src, indices, self._backup)
        self._tgt.remove_ngram(tgt, indices, self._backup)

    def generate_unigram_pairs(self, min_coocc=1, max_coocc=None):
        src_index = self._src._index
        tgt_index = self._tgt._index
        corp_len = len(self._src)

        src_len = len(src_index)
        for i, src_tok in enumerate(src_index):
            # logging
            if i * 100 / src_len < (i + 1) * 100 / src_len:
                logging.info("{0}% done.".format((i+1)*100 / src_len))

            src_occ = src_index[src_tok] 

            possible_tgts = (dict(sorted(self._src_coocc_cache[src_tok][0].items(), key=itemgetter(1), reverse=True)[:20]) if self._coocc_caching else tgt_index)

            logging.debug("{0} - {1}".format(src_tok, len(possible_tgts)))

            for tgt_tok in possible_tgts:
                tgt_occ = tgt_index[tgt_tok]
                coocc = possible_tgts[tgt_tok]
                if (coocc >= min_coocc and (max_coocc is None or coocc <= max_coocc)):
                    cont_table = (coocc, len(src_occ) - coocc, len(tgt_occ) - coocc, corp_len - len(src_occ) - len(tgt_occ) + coocc)
                    yield (((src_tok,), (tgt_tok,)), cont_table)


    def read_from_file(self, f):
        gc.disable()
        logging.info("Reading bicorpus started...")
        c = 1
        for l in f:
            if c % 10000 == 0:
                logging.debug("{0} lines read.".format(c))
            l = l.rstrip("\n")
            if len(l) == 0:
                continue
            src, tgt = l.decode("utf-8").split("\t")
            self.add_sentence_pair((src.split(), tgt.split()))
            c += 1
        gc.enable()
        logging.info("Reading bicorpus done.")
        #from guppy import hpy
        #h = hpy()
        #print h.heap()
        #quit()

    def set_stopwords(self, src, tgt):
        self._src.set_stopwords(src)
        self._tgt.set_stopwords(tgt)

