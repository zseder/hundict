import logging
import gc

from sentence import Sentence
from corpus import Corpus

class BiCorpus:
    def __init__(self, src, tgt, backup=False, tok_coocc_caching=True, int_tokens=False):
        self._src = src
        self._tgt = tgt
        self._backup = backup
        self._coocc_caching = tok_coocc_caching
        if self._coocc_caching:
            self._src_coocc_cache = {}

        self._int_tokens = int_tokens
        if int_tokens:
            self._src_tokens = {} 
            self._tgt_tokens = {}

    def write(self, out):
        for src_sen, tgt_sen in zip(self._src, self._tgt):
            src_str = src_sen.to_str(self._backup)
            tgt_str = tgt_sen.to_str(self._backup)
            out.write(u"{0}\t{1}\n".format(src_str, tgt_str).encode("utf-8"))

    def tokens_to_ints(self, tokens, tokmap):
        ints = []
        for tok in tokens:
            if not tok in tokmap:
                tokmap[tok] = len(tokmap)
            ints.append(tokmap[tok])
        return ints

    def add_sentence_pair(self, pair):
        src, tgt = pair
        if self._int_tokens:
            src = self.tokens_to_ints(pair[0], self._src_tokens)
            tgt = self.tokens_to_ints(pair[1], self._tgt_tokens)
        src_sen = Sentence(src)
        tgt_sen = Sentence(tgt)
        self._src.append(src_sen)
        self._tgt.append(tgt_sen)
        if self._coocc_caching:
            for stok in src_sen:
                for ttok in tgt_sen:
                    try:
                        self._src_coocc_cache[stok].add(ttok)
                    except KeyError:
                        self._src_coocc_cache[stok] = set([ttok])

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
        if self._int_tokens:
            src = tuple(self.convert_tokens_to_int(src, self._src_tokens))
            tgt = tuple(self.convert_tokens_to_int(tgt, self._tgt_tokens))
        indices = self._src.ngram_index(src) & self._tgt.ngram_index(tgt)
        self._src.remove_ngram(src, indices, self._backup)
        self._tgt.remove_ngram(tgt, indices, self._backup)

    def generate_unigram_pairs(self, min_coocc=1, max_coocc=None):
        src_index = self._src._index
        tgt_index = self._tgt._index
        corp_len = len(self._src)

        src_len = len(src_index)
        for i, src_tok in enumerate(src_index):
            if i * 100 / src_len < (i + 1) * 100 / src_len:
                logging.info("{0}0% done.".format(i*10 / src_len))
            logging.debug(src_tok)
            src_occ = src_index[src_tok] 
            possible_tgts = (self._src_coocc_cache[src_tok] if self._coocc_caching else tgt_index)
            for tgt_tok in possible_tgts:
                tgt_occ = tgt_index[tgt_tok]
                coocc = src_occ.intersection(tgt_occ)
                if len(coocc) >= min_coocc and (max_coocc is None or 
                                               len(coocc) <= max_coocc):
                    cont_table = (len(coocc), len(src_occ.difference(tgt_occ)), len(tgt_occ.difference(src_occ)), corp_len - len(src_occ.union(tgt_occ)))
                    yield (((src_tok,), (tgt_tok,)), cont_table)


    @staticmethod
    def read_from_file(f, caching=True):
        src_c = Corpus()
        tgt_c = Corpus()
        bc = BiCorpus(src_c, tgt_c, caching, int_tokens=True)
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
            bc.add_sentence_pair((src.split(), tgt.split()))
            c += 1
        gc.enable()
        logging.info("Reading bicorpus done.")

        return bc
