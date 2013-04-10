from math import log
import logging
from optparse import OptionParser
import sys

from dictionary import Dictionary
from bicorpus import BiCorpus

class DictBuilder:
    def __init__(self, bicorpus, scorer):
        self._bicorpus = bicorpus
        self._scorer = scorer
        self._dict = Dictionary()

        # TODO create options from these
        self.set_bound_multiplier = 10
        self.strdiff = True
        self.ngrams = False
        self.sets = False
        self.sparse_bound = 5
        self.uniset_min = 2
        self.uniset_max = 3

    def filter_mutual_pairs(self, pairs):
        best_src = {}
        best_tgt = {}
        scores = {}
        for ngram_pair in pairs:
            score = pairs[ngram_pair]
            src, tgt = ngram_pair
            scores[ngram_pair] = score
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

    def extend_pair_with_ngrams(self, orig_pair, orig_score, ratio=0.97):
        to_process = set([(orig_pair, orig_score)])
        final = []
        max_score = orig_score
        done = set()
        while True:
            if len(to_process) > 0:
                actual_pair = to_process.pop()
            else:
                break

            actual_pair, actual_score = actual_pair
            if actual_pair in done:
                continue

            src, tgt = actual_pair
            if actual_score / max_score < ratio:
                continue

            src_occ = self._bicorpus._src.ngram_index(src)
            tgt_occ = self._bicorpus._tgt.ngram_index(tgt)
            ngram_indices = src_occ & tgt_occ
            children = self._bicorpus.ngram_pair_neighbours(actual_pair, ngram_indices)
            for child in children:
                child_pair, _, src_changed = child
                if src_changed:
                    table = self._bicorpus.contingency_table(child_pair, tgt_occ_s=tgt_occ)
                else:
                    table = self._bicorpus.contingency_table(child_pair, src_occ_s=src_occ)
                child_score = self.score(table)
                if child_score / max_score > ratio:
                    to_process.add((child_pair, child_score))
                max_score = max(max_score, child_score)
            final.append((actual_pair, actual_score))
            done.add(actual_pair)
        
        if final[-1][1] / orig_score <= 1.0 / ratio:
            return None

        else:
            return final[-1]

    def extend_with_ngrams(self, pairs, scale=1.0):
        def __ngram_pair_parents(ngram_pair):
            parents = []
            src, tgt = ngram_pair
            if len(src) > 1:
                parents.append((src[1:], tgt))
                parents += __ngram_pair_parents((src[1:], tgt))
                parents.append((src[:-1], tgt))
                parents += __ngram_pair_parents((src[:-1], tgt))
            if len(tgt) > 1:
                parents.append((src, tgt[1:]))
                parents += __ngram_pair_parents((src, tgt[1:]))
                parents.append((src, tgt[:-1]))
                parents += __ngram_pair_parents((src, tgt[:-1]))
            return parents
        
        logging.info("Extending dictionary with ngrams started.")
        to_delete = set()
        new_pairs = {}
        orig_pairs = pairs
        status = 0
        for pair in orig_pairs.iterkeys():
            if status * 100 / len(orig_pairs) > (status + 1) * 100 / len(orig_pairs):
                logging.info("{0}% done".format(status * 100 / len(orig_pairs)))
            status += 1
            if pair in to_delete:
                continue

            score = orig_pairs[pair]
            better = self.extend_pair_with_ngrams(pair, score)
            if better is None:
                continue
            else:
                better_pair, better_score = better

                # check if it is a trivial pair
                if (len(better_pair[0]) == len(better_pair[1]) and
                    reduce(lambda x,y: x and y, ((((x,),(y,)) in orig_pairs) for x,y in zip(better_pair[0], better_pair[1])))):
                    continue

                # if there is a better parent, keep that and throw new child away,
                # if there isnt, keep new child and remove worse parents
                is_better_parent = False
                parents = __ngram_pair_parents(pair)
                parents = __ngram_pair_parents(better_pair)
                for parent in parents:
                    if parent in orig_pairs:
                        if orig_pairs[parent] > better_score:
                            is_better_parent = True
                            break
                if not is_better_parent:
                    new_pairs[better_pair] = better_score
                    to_delete |= set(parents)
                    logging.debug("{0} is better than {1}".format(
                        better_pair, parents))

        for p in to_delete:
            if p in orig_pairs:
                del orig_pairs[p]

        logging.info("Extending dictionary with ngrams finished with " +
                    "{0} new pairs.".format(len(new_pairs)))
        return dict(orig_pairs.items() + new_pairs.items())

    def remove_ngram_pairs(self, pairs):
        for pair in pairs:
            score = pairs[pair]
            self._dict[pair] = score
        self._bicorpus.remove_ngram_pairs(pairs.keys())
        self._bicorpus.build_cache()

    def build_low_strdiff_pairs(self):
        if not self.strdiff:
            return

        good_pairs = self._bicorpus.get_low_strdiff_pairs()
        for p in good_pairs:
            cont_table = self._bicorpus.contingency_table(p[0])
            score = self.score(cont_table)
            self._dict[p[0]] = p[1]
            logging.debug("Pair added. ({0})".format(score))
        self._bicorpus.remove_ngram_pairs([p for p in self._dict])

    def build_unigram_pairs(self, bound):
        # get all possible unigram pairs
        unigram_pairs = self._bicorpus.generate_unigram_pairs()

        # filter by sparsity
        filtered_pairs = (pair for pair in unigram_pairs if
                          pair[1][0] + pair[1][1] >= self.sparse_bound and
                          pair[1][0] + pair[1][2] >= self.sparse_bound)


        # count score
        scored_pairs = dict((pair[0], self.score(pair[1])) 
                            for pair in filtered_pairs)

        goods = dict(((pair[0], pair[1]), score)
                 for pair, score in scored_pairs.iteritems() if score >= bound)

        res = dict(self.filter_mutual_pairs(goods))
        logging.info("{0} unigram pairs found at bound {1}".format(len(res), bound))
        return res

    def build_unigram_set_pairs(self, bound):
        """ function to look for pairs, that can be translation pairs only if
        one of the languages contains at least two words. Not ngrams, but
        words, and every time one of them is a translation"""
        logging.info("Searching for unigram set pairs...")
        good_set_pairs = []
        for results in self._bicorpus.generate_unigram_set_pairs(
                min_len=self.uniset_min, max_len=self.uniset_max):
            scores = []
            #collect only good scores
            for src, tgt, table in results:
                score = self.score(table)
                if score < bound:
                    continue
                scores.append((src, tgt, score))
            
            if len(scores) > 0:
                #sort scores and append
                scores.sort(key=lambda x: x[2], reverse=True)
                good_set_pairs.append(scores)
        logging.info("{0} unigram set pairs found at bound {1}".format(
            len(good_set_pairs), bound))
        
        to_remove = []
        for scores in good_set_pairs:
            # keep only the best right now
            src, tgt, score  = scores[0]
            self._dict[(tuple(src), tuple(tgt), True)] = score
            for src_ in src:
                for tgt_ in tgt:
                    to_remove.append((src_, tgt_))

        self._bicorpus.remove_ngram_pairs(to_remove)
        self._bicorpus.build_cache()
        logging.info("Searching for unigram set pairs done")

    def build_iter(self, bound):
        # TODO Cleaning corpus from sentences that contain >=2 hapaxes

        mutual_pairs = self.build_unigram_pairs(bound)

        # extend unigrams to ngrams
        if self.ngrams:
            new_ngram_pairs = self.extend_with_ngrams(mutual_pairs)
        else:
            new_ngram_pairs = mutual_pairs

        # remove pairs that are found to be good and yield them
        self.remove_ngram_pairs(new_ngram_pairs)
        
        # searching for unigram set pairs
        if self.sets:
            self.build_unigram_set_pairs(self.set_bound_multiplier * bound)
        
    def build(self, bound, iters):
        logging.info("Building dictionary started...")

        # searching for low strdiff pairs first
        self.build_low_strdiff_pairs()

        for _iter in xrange(iters):
            logging.info("{0}.iteration started".format(_iter))
            self.build_iter(bound)
            logging.info("iteration finished.")
            bound /= 2.0

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
    src_stopwords = set(punct)
    if options.src_stop:
        src_stopwords |= set(file(options.src_stop).read().rstrip("\n").split("\n"))
    tgt_stopwords = set(punct)
    if options.tgt_stop:
        tgt_stopwords |= set(file(options.tgt_stop).read().rstrip("\n").split("\n"))

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

    bc.remove_ngram_pairs(gold)

    db = DictBuilder(bc, scorer)

    db.build(bound, iters=iters)
    for p in db._dict:
        if len(p) == 2:
            src, tgt = p
            print "{0}\t{1}\t{2}".format(db._dict[p],
                                      " ".join(bc._src.ints_to_tokens(src)),
                                      " ".join(bc._tgt.ints_to_tokens(tgt)),)
        elif len(p) == 3:
            src, tgt, _ = p
            for src_tok in src:
                for tgt_tok in tgt:
                    print "{0}\t{1}\t{2}".format(db._dict[p],
                                      " ".join(bc._src.ints_to_tokens(src_tok)),
                                      " ".join(bc._tgt.ints_to_tokens(tgt_tok)),)
    if rem is not None:
        bc.write(open(rem, "w")) 

if __name__ == "__main__":
    main()

