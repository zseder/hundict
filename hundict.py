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

    def extend_with_ngrams(self, orig_pairs, scale=1.0):
        def __ngram_pair_parents(ngram_pair):
           parents = []
           src, tgt = ngram_pair
           if len(src) > 1:
               parents.append((src[1:], tgt))
               parents.append((src[:-1], tgt))
           if len(tgt) > 1:
               parents.append((src, tgt[1:]))
               parents.append((src, tgt[:-1]))
           return parents
        
        logging.info("Extending dictionary with ngrams started.")
        to_process = dict(orig_pairs)
        final = {}
        while True:
            # if there is no more to process -> stop
            if len(to_process) == 0:
                break
            
            to_process_new = {}
            # extend pair with possible ngrams
            for results_for_pair in self._bicorpus.generate_ngram_pairs(to_process.iteritems()):
                parent = results_for_pair[0]
                old_score = to_process[parent]
                best_for_parent = None
                best_score = old_score * scale
                for new_pair, table in results_for_pair[1:]:
                    new_score = self.score(table)
                    if new_score > best_score:
                        best_for_parent = new_pair
                        best_score = new_score

                if best_for_parent is not None:
                    to_process_new[best_for_parent] = best_score
                    for possible_parent in __ngram_pair_parents(best_for_parent):
                        if possible_parent in final:
                            del final[possible_parent]
                else:
                    final[parent] = old_score

            # clean @final in every iteration
            for pair in final.keys():
                # maybe it's already deleted
                if pair not in final:
                    continue

                possible_parents = __ngram_pair_parents(pair)
                is_better_parent = False
                for pp in possible_parents:
                    if pp in final and final[pp] > final[pair] * scale:
                        is_better_parent = True
                        break

                if is_better_parent:
                    del final[pair]
                    continue
                else:

                    for possible_parent in possible_parents:
                        if possible_parent in final:
                            if final[pair] > final[possible_parent]:
                                del final[possible_parent]
                            else:
                                raise Exception("Better parents should have been removed before")

            to_process = to_process_new

        logging.info("Extending dictionary with ngrams finished.")
        return final
        
    def build(self, bound, iters):
        logging.info("Building dictionary started...")

        # searching for low strdiff pairs first
        good_pairs = self._bicorpus.get_low_strdiff_pairs()
        for p in good_pairs:
            cont_table = self._bicorpus.contingency_table(p[0])
            score = self.score(cont_table)
            if score > bound:
                self._dict[p[0]] = p[1]
                logging.debug("Pair accepted. ({0})".format(score))
            else:
                logging.debug("Pair declined. ({0})".format(score))
        self._bicorpus.remove_ngram_pairs([p for p in self._dict])

        for _iter in xrange(iters):
            logging.info("{0}.iteration started at {1}".format(_iter, time.asctime()))

            # Cleaning corpus from sentences that contain >=2 hapaxes

            # get all possible unigram pairs
            unigram_pairs = self._bicorpus.generate_unigram_pairs()

            # count score
            scored_pairs = ((pair[0], self.score(pair[1]) )for pair in unigram_pairs)

            goods = (((pair[0], pair[1]), score) for pair, score in scored_pairs if score >= bound)
            mutual_pairs = list(self.filter_mutual_pairs(goods))

            # extend unigrams to ngrams
            new_ngram_pairs = self.extend_with_ngrams(mutual_pairs)

            # get context of candidates

            # filter results by a sparse checker

            # remove pairs that are found to be good and yield them
            for pair in new_ngram_pairs:
                score = new_ngram_pairs[pair]
                self._dict[pair] = score
            self._bicorpus.remove_ngram_pairs([k[0] for k in mutual_pairs])
            
            logging.info("iteration finished.")

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
    parser.add_option("", "--iter", dest="iters", help="number of iterations")
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
    punct = set([".", "!", "?", ",", "-", ":", "'", "...", "--", ";", "(", ")", "\""])
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

    #import pickle
    #pickle.dump(bc, open("hunglish.bicorpus.pickle", "w"))
    #quit()
    
    bc.remove_ngram_pairs(gold)

    db = DictBuilder(bc, scorer)

    db.build(bound, iters=iters)
    for p in db._dict:
        src, tgt = p
        print u"{0}\t{1}\t{2}".format(db._dict[p],
                                      " ".join(bc._src.ints_to_tokens(src)),
                                      " ".join(bc._tgt.ints_to_tokens(tgt)),).encode("utf-8")

    if rem is not None:
        bc.write(open(rem, "w")) 

if __name__ == "__main__":
    import cProfile
    #cProfile.run("main()")
    main()
