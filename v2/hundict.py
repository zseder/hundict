import sys
import logging

from corpus_processor import CorpusProcessor, collapse_ngrams_in_corpus


def main():
    input_ = sys.argv[1]
    cp = CorpusProcessor()
    cp.read_bicorp(open(input_))
    top1, top2 = cp.most_freq_words()
    ngram_top1, ngram_top2 = cp.extend_freq(top1, top2)
    collapse_ngrams_in_corpus(cp.c1, ngram_top1)
    collapse_ngrams_in_corpus(cp.c2, ngram_top2)
    cp.dump(open(sys.argv[2], "w"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
