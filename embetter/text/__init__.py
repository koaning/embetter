from embetter.error import NotInstalled

try:
    from embetter.text._sbert import SentenceEncoder
except ModuleNotFoundError:
    SentenceEncoder = NotInstalled("SentenceEncoder", "sbert")

try:
    from embetter.text._s2v import Sense2VecEncoder
except ModuleNotFoundError:
    Sense2VecEncoder = NotInstalled("Sense2VecEncoder", "sbert")


__all__ = ["SentenceEncoder", "Sense2VecEncoder"]
