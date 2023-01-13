from embetter.error import NotInstalled
from embetter.vision._colorhist import ColorHistogramEncoder
from embetter.vision._loader import ImageLoader

try:
    from embetter.vision._torchvis import TimmEncoder
except ModuleNotFoundError:
    TimmEncoder = NotInstalled("TimmEncoder", "vision")


__all__ = ["ImageLoader", "ColorHistogramEncoder", "TimmEncoder"]
