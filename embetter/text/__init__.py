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
    Sense2VecEncoder = NotInstalled("BytePairEncoder", "bpemb")

try:
    from embetter.text._spacy import spaCyEncoder
except ModuleNotFoundError:
    spaCyEncoder = NotInstalled("spaCyEncoder", "spacy")


__all__ = ["SentenceEncoder", "Sense2VecEncoder", "BytePairEncoder", "spaCyEncoder"]
