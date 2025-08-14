from embetter.error import NotInstalled
from embetter.text._model2vec import TextEncoder

try:
    from embetter.text._sbert import SentenceEncoder, MatrouskaEncoder, MatryoshkaEncoder
except ModuleNotFoundError:
    SentenceEncoder = NotInstalled("SentenceEncoder", "sbert")
    MatrouskaEncoder = NotInstalled("MatrouskaEncoder", "sbert")
    MatryoshkaEncoder = NotInstalled("MatryoshkaEncoder", "sbert")

try:
    from embetter.text._s2v import Sense2VecEncoder
except ModuleNotFoundError:
    Sense2VecEncoder = NotInstalled("Sense2VecEncoder", "sense2vec")

try:
    from embetter.text._spacy import spaCyEncoder
except ModuleNotFoundError:
    spaCyEncoder = NotInstalled("spaCyEncoder", "spacy")

try:
    from embetter.text._keras import KerasNLPEncoder
except (ImportError, ModuleNotFoundError):
    KerasNLPEncoder = NotInstalled("KerasNLPEncoder", "keras_nlp")


from embetter.text._lite import LiteTextEncoder, learn_lite_text_embeddings


__all__ = [
    "TextEncoder",
    "SentenceEncoder",
    "MatrouskaEncoder",
    "MatryoshkaEncoder",
    "Sense2VecEncoder",
    "spaCyEncoder",
    "KerasNLPEncoder",
    "LiteTextEncoder",
    "learn_lite_text_embeddings",
]
