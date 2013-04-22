from optparse import OptionParser
from collections import defaultdict

def replace_in_input(input_fn, opt):
    tokens_cnt = [defaultdict(int), defaultdict(int)] # two languages
    for line in get_tokens(input_fn):
        for i, tokens in enumerate(line):
            for t in tokens:
                # t_ = process_token(t)
                tokens_cnt[i][t] += 1
    for line in get_tokens(input_fn):
        out_str = ''
        for i, tokens in enumerate(line):
            for t in tokens:
                if tokens_cnt[i][t] < opt.rare_threshold:
                    out_str += ' _RARE_'
                else:
                    out_str += ' ' + t
            out_str += '\t'
        print out_str.strip().encode('utf8')

def get_tokens(fn):
    f = open(fn)
    for l in f:
        s1, s2 = l.split('\t')
        t1 = tokenize(s1.decode('utf8'))
        t2 = tokenize(s2.decode('utf8'))
        yield (t1, t2)
    f.close()

def tokenize(sen):
    return [i.strip() 
            for i in sen.strip().split(' ') if i.strip()]
    
def create_option_parser():
    parser = OptionParser("usage: %prog [options] input_file")
    parser.add_option("--rare-threshold", dest="rare_threshold", type="int", 
                      default=10, help="replace rare words with common symbol")
    return parser

def main():
    parser = create_option_parser()
    opt, args = parser.parse_args()
    input_fn = args[0]
    replace_in_input(input_fn, opt)

if __name__ == '__main__':
    main()
        

