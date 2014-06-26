import sys
import logging
import random


sample_ratio = 0.05
max_ngrams = 5


def sen_to_ints(s, word_to_int):
    return [word_to_int.setdefault(w, len(word_to_int)) for w in s]


def line_to_bisent(l):
    s1, s2 = l.split("\t")
    return s1.split(), s2.split()


def read_bicorp(f):
    logging.info("Input reading...")
    d1, d2 = {}, {}
    c1, c2 = zip(*[(sen_to_ints(s1, d1), sen_to_ints(s2, d2))
                   for s1, s2 in (line_to_bisent(l) for l in f)])
    logging.info("Input reading done.")
    return c1, c2, d1, d2


def add_to_freq(s, freq, needed=None):
    if freq is None:
        freq = {}
    for start in xrange(len(s)):
        # unigrams are not tuples
        w = s[start]
        if needed is None or w in needed:
            freq[w] = freq.get(w, 0) + 1

        if needed is not None:
            max_len = len(s) - start
            for length in xrange(2, min(max_len + 1, max_ngrams)):
                w = tuple(s[start:start+length])

                if needed is None or w in needed:
                    freq[w] = freq.get(w, 0) + 1


def most_freq(wc, top, drop_last=True, min_freq=10):
    sorted_top = sorted(((w, c) for w, c in wc.iteritems() if c >= min_freq),
                        key=lambda x: x[1], reverse=True)[:top]
    last_freq = sorted_top[-1][1]

    # if drop_last is true, drop lowest frequency ones because of possible ties
    # but drop only when as long as top (don't drop if there are only frequent
    # words)
    res = dict((w, c) for w, c in sorted_top
               if not drop_last or (c != last_freq and len(sorted_top) != top))
    return res


def most_freq_bi(c1, c2, top):
    logging.info("Collecting frequent words...")
    wc1, wc2 = {}, {}
    for s1 in c1:
        add_to_freq(s1, wc1)
    for s2 in c2:
        add_to_freq(s2, wc2)
    top1 = most_freq(wc1, top)
    top2 = most_freq(wc2, top)
    logging.info("Collecting frequent words done.")
    return top1, top2


def add_to_context_freq(s, freq, needed):
    for i in xrange(len(s)):
        for l in xrange(1, min(len(s) - i, max_ngrams)):
            if l == 1:
                w = s[i]
            else:
                w = tuple(s[i:i+l])
            if w in needed:
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


def most_frequent_contexts(c1, c2, wc1, wc2, top):
    cf1, cf2 = {}, {}
    logging.info("Filter most frequent contexts with random sampling...")
    c = 0
    slen = int(round(len(c1) * sample_ratio))
    for s1 in random.sample(c1, slen):
        add_to_context_freq(s1, cf1, wc1)
        c += 1
        if c * 50 / slen > (c - 1) * 50 / slen:
            logging.info("filtering {0}% done".format(c * 50 / slen))
    for s2 in random.sample(c2, slen):
        add_to_context_freq(s2, cf2, wc2)
        c += 1
        if c * 50 / slen > (c - 1) * 50 / slen:
            logging.info("filtering {0}% done".format(c * 50 / slen))
    top1 = most_freq(cf1, top)
    top2 = most_freq(cf2, top)
    logging.info("Filter most frequent contexts with random sampling done.")
    return top1, top2


def total_context_freqs(c1, c2, top1, top2, top):
    # WARNING currently very slow, so returns only multiplied freqs

    logging.info("Get freqs for frequent contexts...")
    tf1, tf2 = {}, {}
    c = 0
    for s1 in c1:
        add_to_freq(s1, tf1, top1)
        c += 1
        if c * 5 / len(c1) > (c - 1) * 5 / len(c1):
            logging.info("collecting {0}% done".format(c * 5 / len(c1)))
    for s2 in c2:
        add_to_freq(s2, tf2, top2)
        c += 1
        if c * 5 / len(c1) > (c - 1) * 5 / len(c1):
            logging.info("collecting {0}% done".format(c * 5 / len(c1)))
    top1 = most_freq(tf1, top)
    top2 = most_freq(tf2, top)
    logging.info("Get freqs for frequent contexts done.")
    return top1, top2


def context_freqs(c1, c2, wc1, wc2, top):
    logging.info("Extending...")
    top1, top2 = most_frequent_contexts(c1, c2, wc1, wc2, top)
    cf1 = dict((k, int(v / sample_ratio))for k, v in top1.iteritems())
    cf2 = dict((k, int(v / sample_ratio))for k, v in top2.iteritems())
    logging.info("Extending done.")
    return cf1, cf2


def merge_needed(old, new, top):
    for ngram in new:
        shorter1 = ngram[1:]
        shorter2 = ngram[:-1]
        if shorter1 in old:
            old[shorter1] -= new[ngram]
        if shorter2 in old:
            old[shorter2] -= new[ngram]
        old[ngram] = new[ngram]
    return most_freq(old, top)


def extend_freq(c1, c2, top1, top2, top, l):
    merged1, merged2 = None, None
    while True:
        cf1, cf2 = context_freqs(c1, c2, top1, top2, top)
        old_keys1 = set(top1.iterkeys())
        old_keys2 = set(top2.iterkeys())
        merged1 = merge_needed(top1, cf1, top)
        merged2 = merge_needed(top2, cf2, top)
        change1 = len(set(merged1.iterkeys()) - old_keys1)
        change2 = len(set(merged2.iterkeys()) - old_keys2)
        logging.info("Extending resulted {0} and {1} new keys".format(
            change1, change2))
        if (change1 + change2 < top / 10000):
            break
    return merged1, merged2


def collapse_ngrams_in_sen(s, ngrams):
    to_collapse = []
    for i in xrange(len(s)):
        for l in xrange(2, len(s) - i + 1):
            ngram = tuple(s[i:i+l])
            if ngram in ngrams:
                to_collapse.append((ngram, i, i + l))

    to_collapse.sort(key=lambda x: ngrams[x[0]])
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


def main():
    input_ = sys.argv[1]
    top = 1000000
    c1, c2, d1, d2 = read_bicorp(open(input_))
    top1, top2 = most_freq_bi(c1, c2, top)
    l = len(c1)
    ngram_top1, ngram_top2 = extend_freq(c1, c2, top1, top2, top, l)
    collapse_ngrams_in_corpus(c1, ngram_top1)
    collapse_ngrams_in_corpus(c2, ngram_top2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
