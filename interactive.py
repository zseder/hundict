import logging

from bicorpus import BiCorpus
from hundict import DictBuilder

global bc

def __choose_bests(l):
    return sorted(l, key=lambda x: x[1], reverse=True)[:10]

def bests_for_src(src):
    return __choose_bests(bc._coocc_cache._cache[src].items())

def bests_for_tgt(tgt):
    return __choose_bests([(src, bc._coocc_cache._cache[src][_tgt]) for src in bc._coocc_cache._cache for _tgt in bc._coocc_cache._cache[src] if _tgt == tgt])

def build():
    db = DictBuilder(bc, DictBuilder.wmi)
    db.build(0.000005, iters=3)
    return db._dict

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(module)s - %(levelname)s - %(message)s")
    bc = BiCorpus(backup=False, int_tokens=False, only_counts=True)
    punct = set([".", "!", "?", ",", "-", ":", "'", "...", "--", ";", "(", ")"])
    bc.set_stopwords(set(file("/home/zseder/Proj/langtools/Data/hungarian_stopwords.sziget").read().decode("utf-8").rstrip("\n").split("\n")) | punct,
                     set(file("/home/zseder/Proj/langtools/Data/english_stopwords").read().decode("utf-8").rstrip("\n").split("\n")) | punct)
    bc.read_from_file(file("/home/zseder/Data/HunglishBicorpus/bi/bi_only_stem"))
