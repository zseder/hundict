import sys
import random
import logging

from corpus_processor import CorpusProcessor


class DictBuilder(object):
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.top = 1000000
        self.sample_ratio = 0.05
        self.min_freq = 10
        self.needed1 = self.build_needed(self.c1)
        self.needed2 = self.build_needed(self.c2)
        self.filter_corp()

    def build_needed(self, c):
        logging.info("Computing needed words...")
        wc = {}
        for s in c:
            for w in s:
                wc[w] = wc.get(w, 0) + 1
        srtd = sorted(((k, v) for k, v in wc.iteritems()
                       if v >= self.min_freq),
                      key=lambda x: x[1], reverse=True)
        needed = set(k for k, v in srtd[:self.top])
        logging.info("Computing needed words done.")
        return needed

    def filter_corp(self):
        logging.info("Filtering corpus...")
        for i in xrange(len(self.c1)):
            s1 = set(self.c1[i])
            s2 = set(self.c2[i])
            common = s1 & s2
            for w in common:
                s1.remove(w)
                s2.remove(w)
            useless1 = s1 - self.needed1
            useless2 = s2 - self.needed2
            for w in useless1:
                s1.remove(w)
            for w in useless2:
                s2.remove(w)
            self.c1[i] = list(s1)
            self.c2[i] = list(s2)
        logging.info("Filtering corpus done.")

    def get_needed_coocc(self):
        co1 = {}
        co2 = {}
        for w in self.needed1:
            co1[w] = {}
        for w in self.needed2:
            co2[w] = {}
        slen = int(round(len(self.c1) * self.sample_ratio))
        for i in random.sample(xrange(len(self.c1)), slen):
            for w1 in self.c1[i]:
                for w2 in self.c2[i]:
                    co1[w1][w2] = co1[w1].get(w2, 0) + 1
                    co2[w2][w1] = co2[w2].get(w1, 0) + 1
        n1 = dict((w1, [w for w, _ in
                        sorted(co1[w1].iteritems(),
                               key=lambda x: x[1], reverse=True)[:3]])
                  for w1 in co1)
        n2 = dict((w2, [w for w, _ in
                        sorted(co2[w2].iteritems(),
                               key=lambda x: x[1], reverse=True)[:3]])
                  for w2 in co2)
        return n1, n2

    def build_coocc(self):
        logging.info("Building coocc...")
        n1, n2 = self.get_needed_coocc()
        self.co1 = {}
        self.co2 = {}
        for w in self.needed1:
            self.co1[w] = {"__sum__": 0.0}
        for w in self.needed2:
            self.co2[w] = {"__sum__": 0.0}
        for i in xrange(len(self.c1)):
            for w1 in self.c1[i]:
                for w2 in self.c2[i]:
                    if w2 in n1[w1]:
                        self.co1[w1][w2] = self.co1[w1].get(w2, 0) + 1
                    self.co1[w1]["__sum__"] += 1
                    if w1 in n2[w2]:
                        self.co2[w2][w1] = self.co2[w2].get(w1, 0) + 1
                    self.co2[w2]["__sum__"] += 1
            if i * 100 / len(self.c1) > (i - 1) * 100 / len(self.c1):
                logging.info("coocc indexing {0}% done".format(
                    i * 100 / len(self.c1)))

    def build_pairs(self):
        c = 0
        total = len(self.co1) + len(self.co2)
        for w1 in self.co1:
            for w2 in self.co1[w1]:
                if w2 == "__sum__":
                    continue
                pc = self.co1[w1][w2] / self.co1[w1]["__sum__"]
                if pc > 0.5:
                    yield w1, w2

            c += 1
            if c * 100 / total > (c - 1) * 100 / total:
                logging.info("Building {0}% done".format(c * 100 / total))

        for w2 in self.co2:
            for w1 in self.co2[w2]:
                if w1 == "__sum__":
                    continue
                pc = self.co2[w2][w1] / self.co2[w2]["__sum__"]
                if pc > 0.5:
                    yield w2, w1

            c += 1
            if c * 100 / total > (c - 1) * 100 / total:
                logging.info("Building {0}% done".format(c * 100 / total))


def main():
    input_ = sys.argv[1]
    cp = CorpusProcessor()
    cp.read_bicorp(open(input_))
    db = DictBuilder(cp.c1, cp.c2)
    itow = [w for w, _ in sorted(cp.wtoi.iteritems(), key=lambda x: x[1])]
    del cp
    db.build_coocc()
    for i1, i2 in db.build_pairs():
        print "{0}\t{1}".format(itow[i1], itow[i2])
    #db.build_pairs()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
