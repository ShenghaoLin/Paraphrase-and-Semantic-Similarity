import re
from functools import partial

NONWORD_REGEX = re.compile("\W")

def with_name(obj, name):
    obj.name = name
    return obj

def named(name):
    return partial(with_name, name=name)

def get_words(tweet):
    return [NONWORD_REGEX.sub("", word.lower()) for word in tweet.split()]

def _windows(arr_like, n):
    l = len(arr_like)
    for i in range(l - n + 1):
        yield tuple(arr_like[i:(i+n)])

def gen_word_ngrams(tweet, n):
    words = get_words(tweet)
    return set(_windows(words, n))

def gen_chars_ngrams(tweet, n):
    words = get_words(tweet)
    result = set()
    for word in words:
        result |= set(_windows(word, n))
    return result

def get_ngram_features(ngram_gen, sent_1, sent_2, n):
    s1 = ngram_gen(sent_1, n)
    s2 = ngram_gen(sent_2, n)
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    return [len(s1), len(s2), len(s1 & s2), len(s1 | s2),
        len(s1 - s2), len(s2 - s1)]

# C1= with_name(partial(get_ngram_features, partial(gen_chars_ngrams, n=1)), "C1")
# C2= with_name(partial(get_ngram_features, partial(gen_chars_ngrams, n=2)), "C2")
#
# V1 = with_name(partial(get_ngram_features, partial(gen_word_ngrams, n=1)), "V1")
# V2 = with_name(partial(get_ngram_features, partial(gen_word_ngrams, n=2)), "V2")

# print(C1)
# print(V2)

if __name__ == "__main__":
    tweet1 = "I have no idea what this is"
    tweet2 = "What the bloody hell is this"
    print(get_ngram_features(gen_word_ngrams, tweet1, tweet2, 1))
    print(get_ngram_features(gen_chars_ngrams, tweet1, tweet2, 2))
