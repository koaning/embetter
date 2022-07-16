import torch
import numpy as np
from embetter.base import EmbetterBase


class TorchVision(EmbetterBase):
    """
    Use a pretrained vision model from TorchVision to generate embeddings.

    You can find a list of available models here:
    https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights

    Usage:

    ```python
    from embetter import TorchVision

    # MobileNet
    TorchVision("pytorch/vision", "mobilenet_v2", "MobileNet_V2_Weights.IMAGENET1K_V2")
    ```
    """

    def __init__(
        self,
        repo_or_dir="pytorch/vision",
        model_name="resnet50",
        weights_name="ResNet50_Weights.IMAGENET1K_V2",
    ):
        self.repo_or_dir = repo_or_dir
        self.weights_name = weights_name
        self.model_name = model_name
        self.weights = torch.hub.load(repo_or_dir, "get_weight", name=weights_name)
        self.model = torch.hub.load(repo_or_dir, model_name, weights=self.weights)
        self.preprocess = self.weights.transforms()

    def transform(self, X, y=None):
        """
        Transforms grabbed images into numeric representations.
        """
        batch = [self.preprocess(x).unsqueeze(0) for x in X]
        return np.array([self.model(x).squeeze(0).detach().numpy() for x in batch])
