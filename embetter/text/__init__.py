from embetter.error import NotInstalled

try:
    from embetter.text._sbert import SentenceEncoder
except ModuleNotFoundError:
    SentenceEncoder = NotInstalled("SentenceEncoder", "sbert")
