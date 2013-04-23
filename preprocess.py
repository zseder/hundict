from optparse import OptionParser
from collections import defaultdict
from string import lower 



def replace_in_input(input_fn, opt):
    tokens_cnt = [defaultdict(int), defaultdict(int)] # two languages
    for line in get_tokens(input_fn):
        for i, tokens in enumerate(line):
            for t in tokens:
                tokens_cnt[i][t] += 1
    named_entities, tokens_cnt = get_named_entities(tokens_cnt, opt)  
    # strings are in lower form

    if opt.ne_replacement == False:
        named_entities = ({}, {})

    tokens_cnt = map_numeric_values(tokens_cnt, opt)
    rare_words = get_rare_words(tokens_cnt, opt)    
    
    #print rare_words 
    for line in get_tokens(input_fn):
        out_str = ''
        for i, tokens in enumerate(line):
            for t_ in tokens:
                t = lower(t_)
                if t in named_entities[i]:
                    out_str += ' _NAMED_ENTITY_'
                elif process_token(t, opt):
                    out_str += ' _NUMERIC_'
                elif t in rare_words[i]:
                    out_str += ' _RARE_'
                else:
                    out_str += ' ' + t
            out_str += '\t'
        print out_str[:-1].encode('utf8')


def get_rare_words(tokens_cnt, opt):
    
    rare_words =  ({}, {})
    if opt.rare_threshold == -1:
        return rare_words
    else:
        for i, tok_cnt in enumerate(tokens_cnt):
            freqs = sorted([(tok, tok_cnt[tok]) for tok in tok_cnt], key=lambda x:x[1], reverse = True)
            thr_ind, limit_freq = opt.rare_threshold, freqs[min(opt.rare_threshold, len(freqs) - 1)]
            
            next_freq = freqs[min(thr_ind + 1, len(freqs) - 1)]
            while limit_freq == next_freq and thr_ind < len(freqs):
                thr_ind += 1
                next_freq = freqs[thr_ind][1]
            
            for item in freqs[thr_ind:]:
                rare_words[i][item[0]] = 1
    return rare_words               


def map_numeric_values(tokens_cnt, opt):

    if opt.num_replacement:
        for tok_cnt in tokens_cnt:
            keys = tok_cnt.keys()
            for tok in keys:
                if process_token(tok, opt):
                    tok_cnt['_NUMERIC_'] = tok_cnt['_NUMERIC_'] + tok_cnt[tok]
                    tok_cnt[tok] = 0
    return tokens_cnt 

def get_named_entities(tokens_cnt, opt):
    

    named_entities = ({}, {})
    for i, tok_cnt in enumerate(tokens_cnt):
            tok_cnt_keys = tok_cnt.keys()
            for tok in tok_cnt_keys:
                if not tok.islower():
                    if lower(tok) not in tok_cnt or float(tok_cnt[tok])/float(tok_cnt[lower(tok)]) > 5:
                        named_entities[i][lower(tok)] = 1
                    tok_cnt[lower(tok)] = tok_cnt[lower(tok)] + tok_cnt[tok]
            for tok in tok_cnt_keys:  
                if not tok.islower():
                    tok_cnt[tok] = 0
    return named_entities, tokens_cnt    


def process_token(t_, opt):
    if opt.num_replacement == False:
        return False
    else:
        string = t_
        num = False
        if string.replace(".", "", 1).replace("-", "", 1).isdigit():
            num = True
        
        return num      


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

    parser.add_option('-s', "--size of vocabulary retained", dest="rare_threshold", type="int",
                      default=-1, help="replace rare words with common symbol")

    parser.add_option('-n', "--numeric-values-replacement", action='store_true', dest='num_replacement',
                     help="replace numeric strings with common symbol", default=False)


    parser.add_option('-c', "--cap/noncap ratio based NE filter", action='store_true', dest='ne_replacement',
                     help='filter named entites based on ration of capitalized/noncapitalized form', default=False ) 

    return parser

def main():
    parser = create_option_parser()
    opt, args = parser.parse_args()
    input_fn = args[0]
    replace_in_input(input_fn, opt)

if __name__ == '__main__':
    main()
        

