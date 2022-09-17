from embetter.error import NotInstalled
from embetter.vision._grab import ImageGrabber
from embetter.vision._colorhist import ColorHistogramEncoder

try:
    from embetter.vision._torchvis import TimmEncoder
except ModuleNotFoundError:
    TimmEncoder = NotInstalled("TimmEncoder", "vision")


__all__ = ["ImageGrabber", "ColorHistogramEncoder", "TimmEncoder"]
