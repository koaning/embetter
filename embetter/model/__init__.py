from embetter.error import NotInstalled
from ._diff import DifferenceClassifier

try:
    from _sbert import SbertLearner
except ModuleNotFoundError:
    Sense2VecEncoder = NotInstalled("SbertLearner", "sentence-tfm")

__all__ = ["DifferenceClassifier", "SbertLearner"]
