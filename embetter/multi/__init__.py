from embetter.error import NotInstalled

try:
    from embetter.multi._clip import ClipEncoder
except ModuleNotFoundError:
    ClipEncoder = NotInstalled("ClipEncoder", "sentence-tfm")
