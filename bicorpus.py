import logging
import gc
from collections import defaultdict
from itertools import combinations

from langtools.utils.stringdiff import levenshtein

from corpus import Corpus

class BiCorpus:
    def __init__(self, backup=False, int_tokens=False):
        self._src = Corpus(backup, int_tokens)
        self._tgt = Corpus(backup, int_tokens)
        self._backup = backup

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

    def create_cache(self):
        if hasattr(self, "_coocc_cache"):
            del self._coocc_cache
        if hasattr(self, "interesting"):
            del self.interesting
        self._coocc_cache = defaultdict(int)
        self.interesting = defaultdict(dict)

    def build_cache(self):
        logging.info("Buildind cache...")
        self.create_cache()
        for sen_i in xrange(len(self._src)):
            if sen_i * 100 / len(self._src) > (sen_i - 1) * 100 / len(self._src):
                logging.debug("{0}% done".format(sen_i * 100/len(self._src)))
            self.add_sentence_pair_to_cache(self._src[sen_i], self._tgt[sen_i])
        self.filter_interesting_pairs()
        logging.info("Buildind cache done")

    def add_sentence_pair_to_cache(self, src, tgt):
        for src_tok in src:
            for tgt_tok in tgt:
                try:
                    self.interesting[src_tok][tgt_tok] += 1
                except KeyError:
                    self.interesting[src_tok][tgt_tok] = 1

    def filter_interesting_pairs(self, max_per_word=10):
        logging.info("Filtering interesting pairs...")
        
        # there's no need for coocc cache in this phase. (should have been deleted
        # earlier?)
        if hasattr(self, "_coocc_cache"):
            del self._coocc_cache

        for src in self.interesting:
            self.interesting[src] = dict(sorted(self.interesting[src].iteritems(), key=lambda x: x[1], reverse=True)[:max_per_word])
        logging.info("Filtering interesting pairs done")

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

            # remove all occurences from @interesting cache
            for src_tok in src:
                for tgt_tok in tgt:
                    try:
                        self.interesting[src_tok][tgt_tok] -= len(indices)
                    except KeyError:
                        pass

        for ngram in src_ngram_to_remove:
            indices = src_ngram_to_remove[ngram]
            self._src.remove_ngram(ngram, indices, self._backup)

        for ngram in tgt_ngram_to_remove:
            indices = tgt_ngram_to_remove[ngram]
            self._tgt.remove_ngram(ngram, indices, self._backup)
        gc.enable() 

        logging.info("Removing pairs done.")

    def get_low_strdiff_pairs(self):
        logging.info("String difference phase started")
        src_index = self._src._index
        tgt_index = self._tgt._index
        for src in self.interesting:
            src_tok = self._src.ints_to_tokens([src])[0].lower()
            for tgt, _ in self.interesting[src].iteritems():
                tgt_tok = self._tgt.ints_to_tokens([tgt])[0].lower()
                ratio = float(len(src_index[src])) / len(tgt_index[tgt])
                if ratio > 3 or ratio < 1/3.0:
                    continue

                #if idiff == 0:
                if src_tok == tgt_tok:
                    logging.debug("{0} added".format(repr((src_tok, tgt_tok))))
                    yield ((src,), (tgt,)), 1.0
                    break

                idiff = levenshtein(src_tok, tgt_tok)
                #sdiff = levenshtein(src_tok, tgt_tok, 1)
                if len(src_tok) >= 5 and len(tgt_tok) >= 5:
                    if idiff == 1:
                        logging.debug("{0} = {1} added".format(repr((src_tok, tgt_tok)), idiff))
                        yield ((src,), (tgt,)), 0.8
                        break
                if len(src_tok) >= 7 and len(tgt_tok) >= 7 and abs(len(tgt_tok) - len(src_tok)) <= 1:
                    if idiff == 2:
                        logging.debug("{0} = {1} added".format(repr((src_tok, tgt_tok)), idiff))
                        yield ((src,), (tgt,)), 0.6
                        break

        logging.info("String difference phase done")

    def generate_unigram_pairs(self, min_coocc=1, max_coocc=None):
        for result in self.generate_unigram_set_pairs(min_coocc, max_coocc, max_len=1):
            src_ngram_set, results_for_src = result
            src_ngram = src_ngram_set.pop()
            for result_for_src in results_for_src:
                tgt_ngram_set, cont_table = result_for_src
                tgt_ngram = tgt_ngram_set.pop()
                yield ((src_ngram, tgt_ngram), cont_table)

    def generate_unigram_set_pairs(self, min_coocc=1, max_coocc=None, min_len=1, max_len=3):
        src_index = self._src._index
        tgt_index = self._tgt._index
        gc.disable()
        src_len = len(src_index)
        for i, src_tok in enumerate(src_index):
            if i * 100 / src_len < (i + 1) * 100 / src_len:
                logging.info("{0}% done.".format((i+1)*100 / src_len))

            src_occ = src_index[src_tok] 

            possible_tgts = self.interesting[src_tok].items()
            sum_ = sum((x[1] for x in possible_tgts))
            sorted_possible_tgts = sorted((x for x in possible_tgts if x[1] >= sum_ / 10), key=lambda x: x[1], reverse=True)[:max_len+2]

            results = []
            for subset_len in xrange(min_len, max_len + 1):
                for tgt_toks in combinations(sorted_possible_tgts, subset_len):
                    tgt_occ = set()
                    if subset_len > 1:
                        # speedup: if we want to match a word for a set of others, we
                        # filter low frequency words
                        if len(src_occ) <= 20:
                            break
                        
                        # creating union of sets
                        # check if there are at least two independent occurences
                        # of every token
                        gain_for_every_word = True
                        for tgt_tok, _ in tgt_toks:
                            tgt_set = tgt_index[tgt_tok]
                            prev_len = len(tgt_occ)
                            tgt_occ |= tgt_set
                            if len(tgt_occ) - prev_len <= 1:
                                gain_for_every_word = False
                                break
                        if not gain_for_every_word:
                            continue

                    else:
                        tgt_occ = tgt_index[tgt_toks[0][0]]
                    coocc = len(src_occ & tgt_occ)
                    if (coocc >= min_coocc and (max_coocc is None or coocc <= max_coocc)):
                        cont_table = self.contingency_table(None, src_occ_s=src_occ, tgt_occ_s=tgt_occ, coocc_c=coocc)
                        results.append( (set([(tgt_tok[0],) for tgt_tok in tgt_toks]), cont_table) )
                    else:
                        break
            yield set([(src_tok,)]), results
        gc.enable()

    def ngram_pair_neighbours(self, pair, ngram_indices=None, max_len=4):
        src, tgt = pair
        if ngram_indices is None:
            src_occ = self._src.ngram_index(src)
            tgt_occ = self._tgt.ngram_index(tgt)
            ngram_indices = src_occ & tgt_occ
        src_neighbours = self._src.ngram_neighbours(src, ngram_indices)
        tgt_neighbours = self._tgt.ngram_neighbours(tgt, ngram_indices)
        all_new_ngram_pairs = []
        if len(src) < max_len:
            for (neighbour, direction), count in src_neighbours:
                new_src_ngram = (src + (neighbour,) if direction == 1 else (neighbour,) + src)
                all_new_ngram_pairs.append(((new_src_ngram, tgt), count, True))
        if len(tgt) < max_len:
            for (neighbour, direction), count in tgt_neighbours:
                new_tgt_ngram = (tgt + (neighbour,) if direction == 1 else (neighbour,) + tgt)
                all_new_ngram_pairs.append(((src, new_tgt_ngram), count, False))
        return all_new_ngram_pairs

    def generate_ngram_pairs(self, previous_ngram_pairs, min_coocc=1):
        for pair in previous_ngram_pairs:
            (src, tgt), _ = pair
            src_occ = self._src.ngram_index(src)
            tgt_occ = self._tgt.ngram_index(tgt)
            ngram_indices = src_occ & tgt_occ
            all_new_ngram_pairs = self.ngram_pair_neighbours(pair, ngram_indices)
            
            results_for_pair = [(src, tgt)]
            for pair, coocc_c, src_changed in all_new_ngram_pairs:
                new_src, new_tgt = pair
                if src_changed:
                    cont_table = self.contingency_table(pair, tgt_occ_s=tgt_occ)
                else:
                    cont_table = self.contingency_table(pair, src_occ_s=src_occ)
                results_for_pair.append(((new_src, new_tgt), cont_table))
            yield results_for_pair

    def contingency_table(self, ngram_pair, **kwargs):
        """
        counts contingency table for a given ngram pair
        all four values of the table can be given independently to
        the method not to be counted twice
        """
        if ngram_pair is not None:
            src, tgt = ngram_pair
        src_occ_s = (kwargs["src_occ_s"] if "src_occ_s" in kwargs else self._src.ngram_index(src))
        tgt_occ_s = (kwargs["tgt_occ_s"] if "tgt_occ_s" in kwargs else self._tgt.ngram_index(tgt))
        
        coocc_c = (kwargs["coocc_c"] if "coocc_c" in kwargs else len((kwargs["coocc_s"] if "coocc_s" in kwargs else src_occ_s & tgt_occ_s)))

        src_occ_c = len(src_occ_s)
        tgt_occ_c = len(tgt_occ_s)
        only_src_c = src_occ_c - coocc_c
        only_tgt_c = tgt_occ_c - coocc_c

        others_c = (kwargs["others_c"] if "others_c" in kwargs else len(self._src) - only_src_c - only_tgt_c - coocc_c)

        return (coocc_c, only_src_c, only_tgt_c, others_c)

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
        self.build_cache()
        gc.enable()
        logging.info("Reading bicorpus done.")

    def set_stopwords(self, src, tgt):
        self._src.set_stopwords(src)
        self._tgt.set_stopwords(tgt)

