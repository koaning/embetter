from embetter.error import NotInstalled
from embetter.vision._grab import ImageGrabber
from embetter.vision._colorhist import ColorHistogram


try:
    from embetter.vision._torchvis import TorchVision
except ModuleNotFoundError:
    TorchVision = NotInstalled("TorchVision", "torch")


__all__ = ["ImageGrabber", "ColorHistogram"]
