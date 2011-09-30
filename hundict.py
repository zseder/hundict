from math import log
import logging
from optparse import OptionParser
import time
import sys

from dictionary import Dictionary
from bicorpus import BiCorpus

class DictBuilder:
    def __init__(self, bicorpus, scorer):
        self._bicorpus = bicorpus
        self._scorer = scorer
        self._dict = Dictionary()

    def filter_mutual_pairs(self, pairs):
        best_src = {}
        best_tgt = {}
        scores = {}
        for pair in pairs:
            ngram_pair, score = pair
            src, tgt = ngram_pair
            scores[pair[0]] = score
            # if already saved, 
            if best_src.has_key(src):
                # save if better
                if best_src[src][0][1] < score:
                    best_src[src] = [(tgt, score)]
                # store if same
                elif abs(best_src[src][0][1] - score) < 1e-7:
                    best_src[src].append((tgt, score))

            # if not saved, save it
            else:
                best_src[src] = [(tgt, score)]

            # same as above but with tgt tokens
            if best_tgt.has_key(tgt):
                if best_tgt[tgt][0][1] < score:
                    best_tgt[tgt] = [(src, score)]
                elif abs(best_tgt[tgt][0][1] - score) < 1e-7:
                    best_tgt[tgt].append((src, score))
            else:
                best_tgt[tgt] = [(src, score)]

        # filter if mutual bests
        for src in best_src.keys():
            # if there are two tgts with same score, skip src
            if len(best_src[src]) != 1:
                continue

            try:
                best_srcs_for_tgt = best_tgt[best_src[src][0][0]]
                # it there are two srcs for tgt, skip it
                if len(best_srcs_for_tgt) != 1:
                    continue
                
                best_src_for_tgt = best_srcs_for_tgt[0]
                if best_src_for_tgt[0] == src:
                    new_ngram_pair = (src, best_src[src][0][0])
                    yield (new_ngram_pair, scores[new_ngram_pair])
            except KeyError, e:
                raise e

    def build(self, bound, iters):
        logging.info("Building dictionary started...")
        for _iter in xrange(iters):
            logging.info("{0}.iteration started at {1}".format(_iter, time.asctime()))

            # Cleaning corpus from sentences that contain >=2 hapaxes

            # get all possible unigram pairs
            unigram_pairs = self._bicorpus.generate_unigram_pairs()

            # count score
            scored_pairs = ((pair[0], self.score(pair[1]) )for pair in unigram_pairs)

            goods = (((pair[0], pair[1]), score) for pair, score in scored_pairs if score >= bound)
            mutual_pairs = self.filter_mutual_pairs(goods)

            # extend unigrams to ngrams

            # get context of candidates

            # filter results by a sparse checker

            # remove pairs that are found to be good and yield them
            for result in mutual_pairs:
                pair, score = result
                self._bicorpus.remove_ngram_pair(pair)
                self._dict[pair] = score
            
            logging.info("done at {0}".format(time.asctime()))

    def score(self, cont_table):
        try:
            return self._scorer(cont_table)
        except ZeroDivisionError:
            return 0.

    @staticmethod
    def pmi(cont_table, weighted=False):
        a,b,c,d = cont_table
        if weighted:
            return float(a) /(a+b+c+d) * log(float(a)*(a+b+c+d)/((a+b)*(a+c)),2)
        else:
            return 1. /(a+b+c+d) * log(float(a)*(a+b+c+d)/((a+b)*(a+c)),2)

    @staticmethod
    def wmi(cont_table):
        return DictBuilder.pmi(cont_table, True)

    @staticmethod
    def dice(cont_table):
        a,b,c,d = cont_table
        return 2.0 * a / (2 * a + b + c)

def create_option_parser():
    parser = OptionParser("usage: %prog [options] input_file bound scorer")
    parser.add_option("-d", "--dict", dest="dict", help="gold dict file")
    parser.add_option("", "--src_stopwords", dest="src_stop", help="src stopwords file")
    parser.add_option("", "--tgt_stopwords", dest="tgt_stop", help="tgt stopwords file")
    parser.add_option("", "--iters", dest="iters", help="number of iterations")
    parser.add_option("-r", "--remaining", dest="remaining", help="output file for remaining corpus")
    parser.add_option("-l", "--loglevel", dest="loglevel", help="logging level. [DEBUG/INFO/WARNING/ERROR/CRITICAL]")
    return parser

def parse_options(parser):
    (options, args) = parser.parse_args()
    input_file = args[0]
    bound = float(args[1])
    scorer = args[2]

    # Default values for options
    # iters
    iters = 3
    if options.iters:
        iters = int(options.iters)

    # stopwords
    punct = set([".", "!", "?", ",", "-", ":", "'", "...", "--", ";", "(", ")"])
    src_stopwords = set()
    if options.src_stop:
        src_stopwords = set(file(options.src_stop).read().decode("utf-8").rstrip("\n").split("\n")) | punct
    tgt_stopwords = set()
    if options.tgt_stop:
        tgt_stopwords = set(file(options.tgt_stop).read().decode("utf-8").rstrip("\n").split("\n")) | punct

    # gold dict
    gold = Dictionary()
    if options.dict:
        gold = Dictionary.read_from_file(file(options.dict))
    
    rem = None
    if options.remaining:
        rem = options.remaining

    if options.loglevel:
        try:
            logging.basicConfig(level=logging.__dict__[options.loglevel], format="%(asctime)s : %(module)s - %(levelname)s - %(message)s")
        except KeyError:
            print "Not a logging level. See(k) help."
            sys.exit(-1)
    return input_file, bound, scorer, iters, src_stopwords, tgt_stopwords, gold, rem

def main():
    optparser = create_option_parser()
    input_file, bound, _scorer, iters, srcstop, tgtstop, gold, rem = parse_options(optparser)
    scorer = getattr(DictBuilder, _scorer)

    backup = rem is not None

    bc = BiCorpus(backup=backup, int_tokens=True)

    bc.set_stopwords(srcstop, tgtstop)

    bc.read_from_file(file(input_file))
    
    for pair in gold:
        bc.remove_ngram_pair(pair)

    db = DictBuilder(bc, scorer)

    db.build(bound, iters=iters)
    for p in db._dict:
        src, tgt = p
        print u"{0}\t{1}\t{2}".format(" ".join(src),
                                      " ".join(tgt),
                                      db._dict[p]).encode("utf-8")

    if rem is not None:
        bc.write(open(rem, "w")) 

if __name__ == "__main__":
    import cProfile
    #cProfile.run("main()")
    main()
