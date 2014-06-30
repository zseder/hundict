import logging
import random


def sen_to_ints(s, word_to_int):
    return [word_to_int.setdefault(w, len(word_to_int)) for w in s]


def line_to_bisent(l):
    s1, s2 = l.split("\t")
    return s1.split(), s2.split()


def collapse_ngrams_in_sen(s, ngrams):
    to_collapse = []
    for i in xrange(len(s)):
        for l in xrange(2, len(s) - i + 1):
            ngram = tuple(s[i:i+l])
            if ngram in ngrams:
                to_collapse.append((ngram, i, i + l))

    to_collapse.sort(key=lambda x: len(x[0]))
    while len(to_collapse) > 0:
        ngram, start, end = to_collapse.pop()
        del s[start:end]
        s.insert(start, ngram)
        new_to_collapse = []
        for i in xrange(len(to_collapse)):
            ngram_, start_, end_ = to_collapse[i]

            # intersecting
            if start_ < end and end_ > start:
                continue

            if start_ >= end:
                # have to change indices because of replace
                l = end - start - 1
                new_to_collapse.append((ngram_, start_ - l, end_ - l))
            else:
                new_to_collapse.append(to_collapse[i])
        to_collapse = new_to_collapse


def collapse_ngrams_in_corpus(c, ngrams):
    for s in c:
        collapse_ngrams_in_sen(s, ngrams)


class CorpusProcessor(object):
    def __init__(self):
        self.max_ngrams = 5
        self.sample_ratio = 0.05
        self.drop_last = True
        self.min_freq = 10
        self.top = 1000000

    def read_bicorp(self, f):
        logging.info("Input reading...")
        d = {}
        c1, c2 = zip(*[(sen_to_ints(s1, d), sen_to_ints(s2, d))
                       for s1, s2 in (line_to_bisent(l) for l in f)])
        logging.info("Input reading done.")
        self.c1 = list(c1)
        self.c2 = list(c2)
        self.wtoi = d
        del c1, c2

    def add_sen_to_freq(self, s, freq, needed=None):
        for start in xrange(len(s)):
            # unigrams are not tuples
            w = s[start]
            if needed is None or w in needed:
                freq[w] = freq.get(w, 0) + 1

            if needed is not None:
                max_len = len(s) - start
                for length in xrange(2, min(max_len + 1, self.max_ngrams)):
                    w = tuple(s[start:start+length])

                    if needed is None or w in needed:
                        freq[w] = freq.get(w, 0) + 1

    def most_freq(self, wc, use_sample_ratio=False):
        min_freq = (self.min_freq * self.sample_ratio if use_sample_ratio
                    else self.min_freq)
        sorted_top = sorted(((w, c) for w, c in wc.iteritems()
                             if c >= min_freq),
                            key=lambda x: x[1], reverse=True)[:self.top]
        if len(sorted_top) == 0:
            return {}
        last_freq = sorted_top[-1][1]

        # if drop_last is true, drop lowest frequency ones because of possible
        #tiesbut drop only when as long as top (don't drop if there are only
        # frequent words)
        res = dict((w, c) for w, c in sorted_top
                   if not self.drop_last or len(sorted_top) != self.top or
                   c != last_freq)
        return res

    def most_freq_words(self):
        logging.info("Collecting frequent words...")
        self.freq1 = {}
        self.freq2 = {}
        for s1 in self.c1:
            self.add_sen_to_freq(s1, self.freq1)
        for s2 in self.c2:
            self.add_sen_to_freq(s2, self.freq2)
        top1 = self.most_freq(self.freq1)
        top2 = self.most_freq(self.freq1)
        logging.info("Collecting frequent words done.")
        return top1, top2

    def add_to_context_freq(self, s, freq, needed, scf):
        for i in xrange(len(s)):
            for l in xrange(1, min(len(s) - i, self.max_ngrams)):
                if l == 1:
                    w = s[i]
                else:
                    w = tuple(s[i:i+l])
                if w in needed and w not in scf:

                    # change int to tuple for sum
                    if l == 1:
                        w = (w, )

                    # create left and right context
                    # prepare for ints and tuples with prev and next_
                    if i > 0:
                        prev = ((s[i-1], ) if type(s[i-1]) == int else s[i-1])
                        wl = prev + w
                        freq[wl] = freq.get(wl, 0) + 1
                    if i < len(s) - 1:
                        next_ = ((s[i+l], ) if type(s[i+l]) == int else s[i+l])
                        wr = w + next_
                        freq[wr] = freq.get(wr, 0) + 1

    def most_frequent_contexts(self, wc1, wc2):
        logging.info("Filter most frequent contexts with random sampling...")
        c = 0
        slen = int(round(len(self.c1) * self.sample_ratio))
        cf1, cf2 = {}, {}

        for s1 in random.sample(self.c1, slen):
            self.add_to_context_freq(s1, cf1, wc1, self.scf1)
            c += 1
            if c * 5 / slen > (c - 1) * 5 / slen:
                logging.info("filtering {0}% done".format(c * 50 / slen))
        self.scf1 |= set(wc1.iterkeys())

        for s2 in random.sample(self.c2, slen):
            self.add_to_context_freq(s2, cf2, wc2, self.scf2)
            c += 1
            if c * 5 / slen > (c - 1) * 5 / slen:
                logging.info("filtering {0}% done".format(c * 50 / slen))
        self.scf2 |= set(wc2.iterkeys())

        top1 = self.most_freq(cf1, True)
        top2 = self.most_freq(cf2, True)
        logging.info("Filter most frequent contexts with random sampling done")
        return top1, top2

    def total_context_freqs(self, top1, top2):
        # WARNING currently very slow, so don't use it

        logging.info("Get freqs for frequent contexts...")
        tf1, tf2 = {}, {}
        c = 0
        for s1 in self.c1:
            self.add_sen_to_freq(s1, tf1, top1)
            c += 1
            if c * 5 / len(self.c1) > (c - 1) * 5 / len(self.c1):
                logging.info("collecting {0}% done".format(
                    c * 50 / len(self.c1)))
        for s2 in self.c2:
            self.add_sen_to_freq(s2, tf2, top2)
            c += 1
            if c * 5 / len(self.c1) > (c - 1) * 5 / len(self.c1):
                logging.info("collecting {0}% done".format(
                    c * 50 / len(self.c1)))
        top1 = self.most_freq(tf1)
        top2 = self.most_freq(tf2)
        logging.info("Get freqs for frequent contexts done.")
        return top1, top2

    def context_freqs(self, needed1, needed2):
        logging.info("Extending...")
        top1, top2 = self.most_frequent_contexts(needed1, needed2)
        cf1 = dict((k, int(v / self.sample_ratio))for k, v in top1.iteritems())
        cf2 = dict((k, int(v / self.sample_ratio))for k, v in top2.iteritems())
        logging.info("Extending done.")
        return cf1, cf2

    def merge_needed(self, old, new):
        for ngram in new:
            shorter1 = ngram[1:]
            shorter2 = ngram[:-1]
            if shorter1 in old:
                old[shorter1] -= new[ngram]
            if shorter2 in old:
                old[shorter2] -= new[ngram]
            old[ngram] = new[ngram]
        return self.most_freq(old)

    def extend_freq(self, top1, top2):
        merged1, merged2 = top1, top2
        # saved contexts
        self.scf1, self.scf2 = set(), set()
        while True:
            cf1, cf2 = self.context_freqs(merged1, merged2)
            old_keys1 = set(merged1.iterkeys())
            old_keys2 = set(merged2.iterkeys())
            merged1 = self.merge_needed(merged1, cf1)
            merged2 = self.merge_needed(merged2, cf2)
            change1 = len(set(merged1.iterkeys()) - old_keys1)
            change2 = len(set(merged2.iterkeys()) - old_keys2)
            logging.info("Extending resulted {0} and {1} new keys".format(
                change1, change2))
            if (change1 + change2 < self.top / 10000):
                break
        return merged1, merged2

    def dump(self, of):
        rev = [w for w, _ in
               sorted(self.wtoi.iteritems(), key=lambda x: x[1])]
        for i in xrange(len(self.c1)):
            s1 = self.c1[i]
            s2 = self.c2[i]
            s1s = " ".join((rev[w] if type(w) == int else
                           "_".join(rev[part] for part in w))
                           for w in s1)
            s2s = " ".join((rev[w] if type(w) == int else
                           "_".join(rev[part] for part in w))
                           for w in s2)
            of.write("{0}\t{1}\n".format(s1s, s2s))
