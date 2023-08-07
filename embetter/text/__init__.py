from embetter.error import NotInstalled

try:
    from embetter.text._sbert import SentenceEncoder
except ModuleNotFoundError:
    SentenceEncoder = NotInstalled("SentenceEncoder", "sentence-tfm")

try:
    from embetter.text._s2v import Sense2VecEncoder
except ModuleNotFoundError:
    Sense2VecEncoder = NotInstalled("Sense2VecEncoder", "sense2vec")

try:
    from embetter.text._bpemb import BytePairEncoder
except ModuleNotFoundError:
    BytePairEncoder = NotInstalled("BytePairEncoder", "bpemb")

try:
    from embetter.text._spacy import spaCyEncoder
except ModuleNotFoundError:
    spaCyEncoder = NotInstalled("spaCyEncoder", "spacy")

try:
    from embetter.text._word2vec import Word2VecEncoder
except ModuleNotFoundError:
    Word2VecEncoder = NotInstalled("Word2VecEncoder", "gensim")


__all__ = [
    "SentenceEncoder",
    "Sense2VecEncoder",
    "BytePairEncoder",
    "spaCyEncoder",
    "Word2VecEncoder",
]
