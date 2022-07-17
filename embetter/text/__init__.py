from embetter.error import NotInstalled

try:
    from embetter.text._sbert import SBERT
except ModuleNotFoundError:
    SBERT = NotInstalled("SBERT", "sbert")
