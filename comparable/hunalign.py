import sys
from tempfile import NamedTemporaryFile as NTO
import subprocess
import itertools

'''s -> (s0,s1), (s1,s2), (s2, s3), ...
see http://docs.python.org/library/itertools.html'''
def pairwise(iterable):
	a, b = itertools.tee(iterable)
	b.next()
	return itertools.izip(a, b)

class Hunalign(object):
    def __init__(self, hunalign, dict_fn, batch_size=100):
        self.hunalign = hunalign
        self.dict_fn = dict_fn
        self.pairs = []
        self.batch_size = batch_size
        self.results = {}

    def add(self, key, src, tgt, src_unstemmed=None, tgt_unstemmed=None):
        if src_unstemmed is not None:
            if len(src_unstemmed) != len(src):
                raise Exception("stemmed and not stemmed have different size")
        if tgt_unstemmed is not None:
            if len(tgt_unstemmed) != len(tgt):
                raise Exception("stemmed and not stemmed have different size")
        src_f = NTO(delete=True)
        tgt_f = NTO(delete=True)
        src_f.write("\n".join(src))
        tgt_f.write("\n".join(tgt))
        src_f.flush()
        tgt_f.flush()
        res_f = NTO(delete=True)
        self.pairs.append((key, src_f, tgt_f, res_f, src, tgt, src_unstemmed, tgt_unstemmed))
        if len(self.pairs) == self.batch_size:
            self.run()
            self.extract_results()
            self.pairs = []

    def run(self):
        # creating batch file
        bf = NTO(delete=True)
        for pair in self.pairs:
            bf.write("{0}\t{1}\t{2}\n".format(
                pair[1].name, pair[2].name, pair[3].name))

        bf.flush()
        options = []
        options.append("-utf")
        options.append("-realign")
        options.append("-batch")
        options.append(self.dict_fn)
        options.append(bf.name)

        self.qualities = {}
        _, stderr = subprocess.Popen([self.hunalign] + options,
                               stderr=subprocess.PIPE).communicate()

        qlines = [l for l in stderr.split("\n") if l.startswith("Quality")]
        for qline in qlines:
            _, res_fn, q = tuple(qline.split("\t"))
            self.qualities[res_fn] = q

        bf.close()

    def extract_results(self):
        for pair in self.pairs:
            key, src_f, tgt_f, align_f, src, tgt, src_un, tgt_un = pair
            src_f.close()
            tgt_f.close()
            ladder = []
            for l in align_f:
                le = l.strip().split("\t")
                try:
                    si = int(le[0])
                    ti = int(le[1])
                    score = le[2]
                    ladder.append((si, ti, score))
                except TypeError:
                    sys.stderr.write("Error in align file:{0}\n".format(
                        repr(le)))
                    continue

            align_fn = align_f.name
            align_f.close()
            if len(ladder) == 0:
                continue

            s = (src if src_un is None else src_un)
            t = (tgt if tgt_un is None else tgt_un)
            outputlines = map( lambda hole:
                (hole[0][2],
                 " ".join(s[int(hole[0][0]):int(hole[1][0])]),
                 " ".join(t[int(hole[0][1]):int(hole[1][1])]))
            ,
                pairwise(ladder)
            )
            #self.results.append((s[prev[0]:actual[0]],
                                 #t[prev[1]:actual[1]],
                                 #ar[2]))
            if key in self.results:
                sys.stderr.write("a key {0} used in twice when added to hunalign\n".format(repr(key)))
                #raise Exception()
            self.results[key] = (outputlines, self.qualities[align_fn])


