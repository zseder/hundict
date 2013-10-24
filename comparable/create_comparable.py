import sys

from hunalign import Hunalign

def read_articles(istream, needed):
    d = {}
    article = []
    title = ""
    for l in istream:
        l = l.strip()
        if l.startswith("%%#PAGE"):
            if len(article) > 0:
                if title in needed:
                    #d[title] = get_first(article)
                    d[title] = article
                article = []
            title = l.decode("utf-8", "ignore").lower().split("page", 1)[1].strip().replace("_", "").replace(" ", "")
        else:
            if len(l) > 0:
                article.append(l)
    if len(article) > 0 and title in needed:
        d[title] = article
    return d

def get_first(article, bound=50):
    words = []
#    useful_sens =  filter(lambda x: len(x.split()) > 5, article)
#    for useful_sen in useful_sens:
    for useful_sen in article:
        # possible table of contents
        if len(useful_sen) < 5:
            break

        useful_words = useful_sen.split()
        words += useful_words
        if len(words) >= bound:
            return words
    return words

def read_pairing(istream, src, tgt):
    pairing = []
    for l in istream:
        le = l.strip().split("\t")
        if len(le) != 4:
            continue

        if le[0] == src and le[2] == tgt:
            pairing.append((le[1].decode("utf-8", "ignore").lower().replace(" ", "").replace("_", ""),
                            le[3].decode("utf-8", "ignore").lower().replace(" ", "").replace("_", "")))
    return pairing

def main():
    pairing = read_pairing(open(sys.argv[3]),
                           sys.argv[4].replace("_", "-"),
                           sys.argv[5].replace("_", "-"))
    #pairing = [("chien", "kutya"), ("chat", "macska")]
    src_needed = set([a for a, _ in pairing])
    tgt_needed = set([a for _, a in pairing])
    src_articles = read_articles(open(sys.argv[1]), src_needed)
    tgt_articles = read_articles(open(sys.argv[2]), tgt_needed)
    hunalign = Hunalign(sys.argv[6], sys.argv[7], 100)
    for src, tgt in pairing:
        if src not in src_articles or tgt not in tgt_articles: continue
        src_art = src_articles[src]
        tgt_art = tgt_articles[tgt]

        src_art = [sen for sen in src_art 
                   if len([w for w in sen.split() if w.isalpha()]) >= 4 and 
                   len([w for w in sen.split() if w.isalpha()]) <= 20 and
                  sen.split()[-1] in [".", "?", "!"]]
        tgt_art = [sen for sen in tgt_art 
                   if len([w for w in sen.split() if w.isalpha()]) >= 4 and 
                   len([w for w in sen.split() if w.isalpha()]) <= 20 and
                  sen.split()[-1] in [".", "?", "!"]]
        if len(src_art) < 2 or len(tgt_art) < 2: continue
        hunalign.add((src, tgt),src_art, tgt_art)

        #if len(tgt_art) == 0: continue
        #ratio = float(len(src_art)) / len(tgt_art)
        #if ratio > 0.5 and ratio < 2.0:
            #print "{0}\t{1}".format(" ".join(src_art), " ".join(tgt_art))

    for docpair in hunalign.results:
        src, tgt = docpair
        senpairs, q = hunalign.results[docpair]
        sys.stdout.write(u"%%#PAIRING {0} {1} {2} {3}\n".format(
            src, tgt, q, len(senpairs)).encode("utf-8"))
        for senpair in senpairs:
            (score, src_sens, tgt_sens) = senpair
            sys.stdout.write("{0}\t{1}\t{2}\n".format(
                src_sens, tgt_sens, score))

if __name__ == "__main__":
    main()
