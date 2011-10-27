import logging
import gc
from collections import defaultdict

from corpus import Corpus
from coocc_cache import CooccCache

class BiCorpus:
    def __init__(self, backup=False, int_tokens=False):
        self._src = Corpus(backup, int_tokens)
        self._tgt = Corpus(backup, int_tokens)
        self._backup = backup
        self._coocc_cache = CooccCache()

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
        i = len(self._src) - 1
        self._coocc_cache.add_sentence_pair((self._src[i], self._tgt[i]), i)

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

    def remove_ngram_pairs(self, pairs):
        """
        this method removes ngram pairs from corpora
        input is a list because if it were only a pair,
        indices can get corrupted
        cause:
          - while they are ngrams, ngram occurences have to be removed
            from cache, not all per token
          - if one token is removed, but has to be removed from the same
            sentence because of another index, it cannot be done
        """
        if len(pairs) == 0:
            return
        logging.info("Removing found pairs")
        gc.disable()
        src_ngram_to_remove = defaultdict(set)
        tgt_ngram_to_remove = defaultdict(set)
        for pair in pairs:
            src, tgt = pair
            indices = self._src.ngram_index(src) & self._tgt.ngram_index(tgt)
            if len(indices) > 0:
                src_ngram_to_remove[src] |= indices
                tgt_ngram_to_remove[tgt] |= indices

        for ngram in src_ngram_to_remove:
            indices = src_ngram_to_remove[ngram]
            self._src.remove_ngram(ngram, indices, self._backup)

        for ngram in tgt_ngram_to_remove:
            indices = tgt_ngram_to_remove[ngram]
            self._tgt.remove_ngram(ngram, indices, self._backup)
        
        # build up coocc_cache again. faster than maintaining it
        logging.info("Building up coocc cache")
        self._coocc_cache = CooccCache()
        for i in xrange(len(self._src)):
            src_sen = self._src[i]
            tgt_sen = self._tgt[i]
            self._coocc_cache.add_sentence_pair((src_sen, tgt_sen), i)
        logging.info("cache built")
        self._coocc_cache.filter()
        gc.enable()
        logging.info("Removing pairs done.")

    def generate_unigram_pairs(self, min_coocc=1, max_coocc=None):
        src_index = self._src._index
        tgt_index = self._tgt._index
        corp_len = len(self._src)
        
        self._coocc_cache.filter()
        gc.disable()

        src_len = len(src_index)
        for i, src_tok in enumerate(src_index):
            # logging
            if i * 100 / src_len < (i + 1) * 100 / src_len:
                logging.info("{0}% done.".format((i+1)*100 / src_len))

            src_occ = src_index[src_tok] 

            possible_tgts = self._coocc_cache.possible_pairs(src_tok)

            logging.debug(u"{0} - {1}".format(src_tok, len(possible_tgts)).encode("utf-8"))

            for tgt_tok in possible_tgts:
                tgt_occ = tgt_index[tgt_tok]
                coocc = self._coocc_cache.coocc_count((src_tok, tgt_tok))
                if (coocc >= min_coocc and (max_coocc is None or coocc <= max_coocc)):
                    cont_table = (coocc, len(src_occ) - coocc, len(tgt_occ) - coocc, corp_len - len(src_occ) - len(tgt_occ) + coocc)
                    yield (((src_tok,), (tgt_tok,)), cont_table)
        gc.enable()


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
            
            #if no-token sentence -> skip it
            if len(src) == 0 or len(tgt) == 0:
                continue

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

