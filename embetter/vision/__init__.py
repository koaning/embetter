from embetter.error import NotInstalled

try:
    from embetter.vision._torchvis import TorchVision
except ModuleNotFoundError:
    TorchVision = NotInstalled("TorchVision", "torch")
