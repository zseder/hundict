# TODO
# maybe dictionary can be stored as a word tree
# if performance is an issue

class Dictionary:
    def __init__(self):
        self._dict = {}

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)
    def __reversed__(self):
        return reversed(self._dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __contains__(self, item):
        return item in self._dict

    @classmethod
    def read_from_file(cls, f):
        d = Dictionary()
        for l in f:
            l = l.strip().decode("utf-8").split("\t")
            if len(l) == 3:
                try:
                    score = float(l[0])
                except IndexError:
                    score = 0.
                except ValueError:
                    score = 0.
                remaining = l[1:]
            elif len(l) == 2:
                remaining = l
            src = tuple(remaining[0].split())
            tgt = tuple(remaining[1].split())
            d[(src, tgt)] = score
        return d

