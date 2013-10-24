import os
import sys

import nltk

from yalign.input_conversion import text_to_document
from yalign.yalignmodel import YalignModel

from create_comparable import read_pairing, read_articles

def write_plaintext(stream, pairs):
    for a, b in pairs:
        stream.write(a.to_text())
        stream.write('\n')
        stream.write(b.to_text())
        stream.write('\n')

def main():
    output_format = "plaintext"
    lang_a = sys.argv[1]
    lang_b = sys.argv[2]
    model_path = os.path.abspath(sys.argv[3])
    nltk.data.path += [model_path]
    model = YalignModel.load(model_path)

    pairing = read_pairing(open(sys.argv[4]), lang_a, lang_b)
    src_needed = set([a for a, _ in pairing])
    tgt_needed = set([a for _, a in pairing])
    src_articles = read_articles(open(sys.argv[5]), src_needed)
    tgt_articles = read_articles(open(sys.argv[6]), tgt_needed)
    for src, tgt in pairing:
        try:
            text_a = "\n".join(src_articles[src])
            text_b = "\n".join(tgt_articles[tgt])
            document_a = text_to_document(text_a, lang_a)
            document_b = text_to_document(text_b, lang_b)
            pairs = model.align(document_a, document_b)
            sys.stderr.write(u"{0} pairs in {1}-{2}\n".format(len(pairs), src, tgt).encode("utf-8"))

            write_plaintext(sys.stdout, pairs)
        except KeyError:
            sys.stderr.write(u"KeyError with {0}-{1}\n".format(src, tgt).encode("utf-8"))
            continue

if __name__ == "__main__":
    #main()
    import cProfile
    cProfile.run("main()")
