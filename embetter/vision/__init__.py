from embetter.error import NotInstalled
from embetter.vision._grab import ImageGrabber


try:
    from embetter.vision._torchvis import TorchVision
except ModuleNotFoundError:
    TorchVision = NotInstalled("TorchVision", "torch")


__all__ = ["ImageGrabber"]
