from embetter.error import NotInstalled
from embetter.vision._loader import ImageLoader
from embetter.vision._colorhist import ColorHistogramEncoder

try:
    from embetter.vision._torchvis import TimmEncoder
except ModuleNotFoundError:
    TimmEncoder = NotInstalled("TimmEncoder", "vision")


__all__ = ["ImageLoader", "ColorHistogramEncoder", "TimmEncoder"]
